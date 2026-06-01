import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.elementwise_binary_impl import elementwise_binary_bpr, elementwise_binary_rrr, elementwise_binary_rpr
from ..utils import ElementwiseBinaryOpType, Schedule

class Elementwise_Binary(vOp):
    r"""
    Binary elementwise dispatcher for rank-3 logical tensors ``[S, C, D]``.

    This operator dispatches to a binary elementwise implementation based on the
    pair of input formats ``(x._format, y._format)``. The logical output shape:

    - keeps the ``S`` axis from the runtime context (``ctx.max_num_pages``), and
    - follows broadcasting over the ``(C, D)`` axes.

    Scalar parameters ``alpha`` and ``beta`` can be used by certain binary
    operations (e.g. an ``axpby``-style op).

    Attributes
    ----------
    _impl_map : Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]]
        Dispatch table keyed by ``(x_format, y_format)``. Each entry maps to
        ``(callable_impl, resolved_output_format)``.

    alpha : float
        Scalar parameter used by some ops. Default is ``1.0``.

    beta : float
        Scalar parameter used by some ops. Default is ``1.0``.

    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.

    op_type : Optional[ElementwiseBinaryOpType]
        The operator type used by the implementation.

    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.

    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer that stores the binary result.
    """

    # Implementation dispatch table: keyed by (x_format, y_format).
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[Tuple[FORMAT, FORMAT], FORMAT] = {
        (FORMAT.RAGGED,  FORMAT.RAGGED): (FORMAT.RAGGED),
        (FORMAT.BATCHED, FORMAT.PAGED):  (FORMAT.RAGGED),
        (FORMAT.RAGGED,  FORMAT.PAGED):  (FORMAT.RAGGED),
        # Add more pairs as needed.
    }

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.impl: Optional[Callable] = None
        self.op_type: Optional[ElementwiseBinaryOpType] = None
        self.alpha = alpha
        self.beta = beta
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W
    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate inputs, select implementation, allocate the output buffer,
        and return a ``vTensor`` view with the resolved output format.

        The dispatcher:

        - checks that ``x`` and ``y`` are rank-3 tensors of shape ``[S, C, D]``
        - enforces broadcastability on the ``C`` and ``D`` dimensions
        - selects an implementation using ``(x._format, y._format)``
        - allocates an output buffer with shape ``[S_ctx, C_out, D_out]`` where

          .. math::

             C_{\text{out}} = \max(C_x, C_y), \quad
             D_{\text{out}} = \max(D_x, D_y),

          and ``S_ctx = ctx.max_num_pages``.

        Parameters
        ----------
        x : vTensor
            Left-hand input tensor, rank-3, with logical shape ``[S, C, D]``.

        y : vTensor
            Right-hand input tensor, rank-3, with logical shape ``[S, C, D]``.

        ctx : Context
            Execution context providing the runtime ``S`` (``ctx.max_num_pages``)
            and auxiliary-memory accounting.

        Returns
        -------
        vTensor
            A ``vTensor`` view wrapping the allocated output buffer, using the
            resolved output format from the dispatch table.

        Raises
        ------
        AssertionError
            If types are not ``vTensor``, if ranks are not 3, if ``C``/``D``
            are not broadcastable, if formats are unsupported, or if devices
            of ``x`` and ``y`` do not match.
        """
        prefix = self._prefix()

        # Type checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # Rank & basic shape checks
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs [S, C, D]; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )

        # Broadcastability on C/D
        assert (x.shape[1] == y.shape[1] or x.shape[1] == 1 or y.shape[1] == 1), (
            f"{prefix}dim-1 not broadcastable: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"
        )
        assert (x.shape[2] == y.shape[2] or x.shape[2] == 1 or y.shape[2] == 1), (
            f"{prefix}dim-2 not broadcastable: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"
        )

        # Dispatch
        x_fmt, y_fmt = x._format, y._format
        key = (x_fmt, y_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}). "
            f"Available: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[key]

        # Device consistency
        assert x.device == y.device, (
            f"{prefix}x and y must be on the same device "
            f"(x.device={x.device}, y.device={y.device})"
        )

        # Broadcasted output (C, D)
        C_out = max(x.shape[1], y.shape[1])
        D_out = max(x.shape[2], y.shape[2])

        # Allocate output buffer on x.device with x.dtype
        # S = ctx.max_num_pages
        self.output_buffer = as_vtensor(torch.empty(
            (0, C_out, D_out),
            device=x.device,
            dtype=x.dtype,
        ), self.output_format, tensor_id=len(ctx.tensor_list)  # Assign a new tensor_id based on current tensor count
        )
        # Track auxiliary memory and graph structure in the context
        ctx.tensor_list.append(self.output_buffer)  # Track the output buffer in the context
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # Map the output tensor to this operation
        ctx.op_list.append(self)  # Track this operation in the context
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])  # Map this op to its input tensors
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # Map this op to its output tensor

        return self.output_buffer
        

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, y: torch.Tensor, ctx: Context) -> torch.Tensor:
        r"""
        Execute the selected binary elementwise implementation into the internal
        output buffer and return it.

        Expected implementation signature::

            impl(x, y, output, op_type, alpha, beta, ctx)

        Parameters
        ----------
        x : torch.Tensor
            Left-hand input tensor on the same device as ``y`` and the output
            buffer.

        y : torch.Tensor
            Right-hand input tensor on the same device as ``x`` and the output
            buffer.

        ctx : Context
            Execution context passed through to the underlying implementation.

        Returns
        -------
        torch.Tensor
            The output tensor stored in ``self.output_buffer``.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called (no implementation or buffer),
            or if there is a device mismatch between inputs and output.
        """
        assert False, "Elementwise_Binary.execute should not be called directly; it is meant to be invoked by the generated Triton kernel. If you see this error, it likely means the kernel was not generated or called correctly."
        # prefix = self._prefix()
        # assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        # assert self.output_buffer is not None, f"{prefix}output buffer is None; did profile() run?"
        # assert x.device == y.device == self.output_buffer.device, (
        #     f"{prefix}device mismatch: "
        #     f"x={x.device}, y={y.device}, o={self.output_buffer.device}"
        # )

        # self.impl(x, y, self.output_buffer, self.op_type, self.alpha, self.beta, ctx)
        # return self.output_buffer


class Maximum(Elementwise_Binary):
    r"""
    Elementwise maximum between two tensors.

    This operator computes the pointwise maximum:

    .. math::

        \operatorname{out}(x, y) = \max(x, y)

    Broadcasting over the ``(C, D)`` axes is supported as described in
    :class:`Elementwise_Binary`.

    Parameters
    ----------
    alpha : float, optional
        Scalar parameter forwarded to the binary kernel. It is not used
        by the maximum operation itself. Default is ``1.0``.

    beta : float, optional
        Scalar parameter forwarded to the binary kernel. It is not used
        by the maximum operation itself. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Maximum


class Minimum(Elementwise_Binary):
    r"""
    Elementwise minimum between two tensors.

    This operator computes the pointwise minimum:

    .. math::

        \operatorname{out}(x, y) = \min(x, y)

    Broadcasting over the ``(C, D)`` axes is supported as described in
    :class:`Elementwise_Binary`.

    Parameters
    ----------
    alpha : float, optional
        Scalar parameter forwarded to the binary kernel. It is not used
        by the minimum operation itself. Default is ``1.0``.

    beta : float, optional
        Scalar parameter forwarded to the binary kernel. It is not used
        by the minimum operation itself. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Minimum
        

class Add(Elementwise_Binary):
    r"""
    Affine combination of two tensors.

    This operator computes a weighted sum of the two inputs:

    .. math::

        \operatorname{out}(x, y) = \alpha x + \beta y

    With the defaults :math:`\alpha = 1` and :math:`\beta = 1`, this
    reduces to standard elementwise addition:

    .. math::

        \operatorname{out}(x, y) = x + y

    Broadcasting over the ``(C, D)`` axes is supported as described in
    :class:`Elementwise_Binary`.

    Parameters
    ----------
    alpha : float, optional
        Scalar multiplier for :math:`x`. Default is ``1.0``.

    beta : float, optional
        Scalar multiplier for :math:`y`. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Add
        

class Multiply(Elementwise_Binary):
    r"""
    Elementwise product of two tensors.

    This operator computes the pointwise product:

    .. math::

        \operatorname{out}(x, y) = x \cdot y

    Broadcasting over the ``(C, D)`` axes is supported as described in
    :class:`Elementwise_Binary`.

    Parameters
    ----------
    alpha : float, optional
        Scalar parameter forwarded to the binary kernel. It is not used
        by the pure multiplication operation itself. Default is ``1.0``.

    beta : float, optional
        Scalar parameter forwarded to the binary kernel. It is not used
        by the pure multiplication operation itself. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Mul

