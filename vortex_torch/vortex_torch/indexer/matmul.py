import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from ..utils import Schedule
from .triton_kernels.mv_impl import mv_bpr
from .triton_kernels.matmul_impl import mm_bpr, mm_rrr, mm_rpr


class GeMV(vOp):
    r"""
    General matrix-vector multiplication (GEMV) dispatcher.

    This operator computes a *piecewise* batched matrix-vector product.
    Let

    .. math::

        X \in \mathbb{R}^{B \times 1 \times D}, \qquad
        Y \in \mathbb{R}^{S_{\text{pack}} \times 1 \times D},

    where the ``S``-axis of :math:`Y` is a concatenation of batch-wise
    segments

    .. math::

        S_{\text{pack}} = \sum_{i=0}^{B-1} S_i, \qquad
        Y =
        \begin{bmatrix}
            Y_0 \\
            Y_1 \\
            \vdots \\
            Y_{B-1}
        \end{bmatrix},

    with

    .. math::

        Y_i \in \mathbb{R}^{S_i \times 1 \times D}, \qquad
        X_i = X[i, 0, :] \in \mathbb{R}^{1 \times D}.

    For each batch index :math:`i \in \{0,\dots,B-1\}`, we define

    .. math::

        O_i = Y_i X_i^{\mathsf{T}} \in \mathbb{R}^{S_i \times 1 \times 1},

    and the overall output is the concatenation

    .. math::

        O =
        \begin{bmatrix}
            O_0 \\
            O_1 \\
            \vdots \\
            O_{B-1}
        \end{bmatrix}
        \in \mathbb{R}^{S_{\text{pack}} \times 1 \times 1}.

    In the runtime, :math:`S_{\text{pack}}` is given by
    ``ctx.max_num_pages`` and the dispatch is keyed by the pair of input
    formats ``(x_format, y_format)``.

    Attributes
    ----------
    _impl_map : Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]]
        Dispatch table keyed by ``(x_format, y_format)``. Each entry maps
        to ``(callable_impl, resolved_output_format)``.

    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.

    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.

    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer of shape ``[S_pack, 1, 1]``.
    """

    # Implementation dispatch table: keyed by (x_format, y_format).
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[Tuple[FORMAT, FORMAT], FORMAT] = {
        (FORMAT.BATCHED, FORMAT.PAGED): FORMAT.RAGGED,
        # Extend with more pairs as needed.
    }

    def __init__(self):
        super().__init__()
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W
    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate inputs, select an implementation, allocate the output buffer,
        and return an :func:`as_vtensor` view with the resolved format.

        The method enforces the logical shapes

        - ``x``: ``[B, 1, D]``
        - ``y``: ``[S_pack, 1, D]``

        and checks that the last dimensions match. The output buffer is
        allocated with shape ``[S_pack, 1, 1]``, where ``S_pack`` is taken
        from the runtime context as ``ctx.max_num_pages``.
        """
        prefix = self._prefix()

        # Type checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # Rank/shape checks
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        assert x.shape[1] == 1, f"{prefix}expected x.shape[1] == 1, got {tuple(x.shape)}"
        assert y.shape[1] == 1, f"{prefix}expected y.shape[1] == 1, got {tuple(y.shape)}"
        assert x.shape[2] == y.shape[2], (
            f"{prefix}last dimension mismatch: x.shape[2]={x.shape[2]} vs y.shape[2]={y.shape[2]}"
        )

        # Dispatch
        x_fmt, y_fmt = x._format, y._format
        key = (x_fmt, y_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}). "
            f"Available: {list(self._impl_map.keys())}"
        )
        # self.impl, self.output_format = self._impl_map[key]
        self.output_format = self._impl_map[key]
        # Allocate output buffer on x.device/x.dtype
        #S_out = ctx.max_num_blocks        # logical "S_pack" per runtime
        self.output_buffer = as_vtensor(torch.empty(
            (0, 1, 1),
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
        Launch the selected GEMV implementation into the internal output buffer.

        Expected kernel signature::

            impl(x, y, output, ctx)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor corresponding to the batched vector(s), with shape
            ``[B, 1, D]`` and on the same device as ``y`` and the output.

        y : torch.Tensor
            Input tensor corresponding to the packed matrix rows, with shape
            ``[S_pack, 1, D]`` and on the same device as ``x`` and the output.

        ctx : Context
            Execution context passed through to the underlying implementation.

        Returns
        -------
        torch.Tensor
            The output tensor stored in ``self.output_buffer`` with shape
            ``[S_pack, 1, 1]``.
        """
        assert False, "GeMV.execute is not implemented yet. Please implement the kernel and then enable this code."
        # prefix = self._prefix()
        # assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        # assert self.output_buffer is not None, f"{prefix}output buffer is None; did profile() run?"
        # assert x.device == y.device == self.output_buffer.device, (
        #     f"{prefix}device mismatch: "
        #     f"x={x.device}, y={y.device}, o={self.output_buffer.device}"
        # )

        # self.impl(x, y, self.output_buffer, ctx)
        # return self.output_buffer



# ------------------------------ GeMM ------------------------------ #
class GeMM(vOp):
    r"""
    General matrix-matrix multiplication (GeMM) dispatcher.

    Logically this computes, for each logical ``S``-slice, a matrix-matrix
    product

    .. math::

        O[s] = Y[s] X[s]^{\mathsf{T}}, \quad s = 0, \dots, S-1,

    with slice-wise shapes

    .. math::

        X[s] \in \mathbb{R}^{N_x \times K}, \quad
        Y[s] \in \mathbb{R}^{N_y \times K}, \quad
        O[s] \in \mathbb{R}^{N_y \times N_x}.

    In the packed 3D representation used by this dispatcher:

    - ``Y`` has logical shape ``[S, N_y, K]``.
    - ``X`` has logical shape ``[L_x, N_x, K]``, where the leading
      dimension :math:`L_x` can represent **either**:

      * a batch axis :math:`B` (when ``x_format == FORMAT.BATCHED``), or
      * the same ``S`` axis as ``Y`` (when ``x_format`` is ragged/paged and
        already laid out per-page).

      This is why the code comments use ``X: [B/S, N_x, K]``: the first
      dimension is interpreted as either a batch size :math:`B` or an
      ``S``-like logical page index, depending on the format.

    - The output tensor ``O`` has logical shape ``[S, N_y, N_x]``.

    At runtime, the logical ``S`` is taken from ``ctx.max_num_pages``, and
    dispatch is keyed by the pair of tensor formats ``(x_format, y_format)``.

    Attributes
    ----------
    _impl_map : Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]]
        Dispatch table keyed by ``(x_format, y_format)``. Each entry maps to
        ``(callable_impl, resolved_output_format)``.

    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.

    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.

    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer of shape ``[S, N_y, N_x]``.
    """

    # Implementation dispatch table: keyed by (x_format, y_format).
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[Tuple[FORMAT, FORMAT], FORMAT] = {
        (FORMAT.BATCHED, FORMAT.PAGED): (FORMAT.RAGGED),
        (FORMAT.RAGGED, FORMAT.RAGGED): (FORMAT.RAGGED),
        (FORMAT.RAGGED, FORMAT.PAGED):  (FORMAT.RAGGED),
        # Extend with more pairs as needed.
    }

    def __init__(self):
        super().__init__()
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate inputs, select implementation, allocate the output buffer,
        and return an :func:`as_vtensor` view with the resolved format.

        The method enforces that both inputs are rank-3 tensors and that the
        inner dimension :math:`K` matches:

        - ``x``: ``[B_or_S, N_x, K]``

          *When* ``x_format == FORMAT.BATCHED``, the leading dimension is a
          batch size :math:`B`. For ragged/paged formats, it may conceptually
          coincide with :math:`S`.

        - ``y``: ``[S, N_y, K]``

        The output buffer is allocated with shape ``[S, N_y, N_x]``, where
        ``S`` is taken from the runtime context as ``ctx.max_num_pages``.

        Parameters
        ----------
        x : vTensor
            Right-hand operand (transposed in the mathematical view), with
            shape ``[B_or_S, N_x, K]`` and a format participating in the
            ``(x_format, y_format)`` dispatch.

        y : vTensor
            Left-hand operand with shape ``[S, N_y, K]``.

        ctx : Context
            Execution context providing ``ctx.max_num_pages`` for the logical
            ``S`` dimension and tracking auxiliary memory.

        Returns
        -------
        vTensor
            A ``vTensor`` view wrapping the allocated output buffer with the
            resolved output format.

        Raises
        ------
        AssertionError
            If types are not ``vTensor``, ranks are not 3, the inner
            dimensions :math:`K` do not match, or there is no implementation
            for the pair ``(x._format, y._format)``.
        """
        prefix = self._prefix()

        # Type checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # Rank/shape checks
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        # K must match
        assert x.shape[2] == y.shape[2], (
            f"{prefix}last dimension mismatch: x.shape[2]={x.shape[2]} vs y.shape[2]={y.shape[2]}"
        )

        # Dispatch
        x_fmt, y_fmt = x._format, y._format
        key = (x_fmt, y_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}). "
            f"Available: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[key]

        # Output logical sizes: Ny x Nx
        Ny, Nx = y.shape[1], x.shape[1]

        # Allocate output buffer on x.device/x.dtype
        self.output_buffer = as_vtensor(torch.empty(
            (0, Ny, Nx),
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
        Launch the selected GeMM implementation into the internal buffer.

        Expected kernel signature::

            impl(x, y, output, ctx)

        Parameters
        ----------
        x : torch.Tensor
            Right-hand operand (transposed in the mathematical view), with
            shape ``[B_or_S, N_x, K]`` on the same device as ``y`` and the
            output buffer.

        y : torch.Tensor
            Left-hand operand with shape ``[S, N_y, K]`` on the same device
            as ``x`` and the output buffer.

        ctx : Context
            Execution context passed through to the underlying implementation.

        Returns
        -------
        torch.Tensor
            The output tensor stored in ``self.output_buffer`` with shape
            ``[S, N_y, N_x]``.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called (no implementation or
            buffer), or if there is a device mismatch between ``x``, ``y``
            and the output buffer.
        """
        assert False, "GeMM.execute is not implemented yet. Please implement the kernel and then enable this code."
        # prefix = self._prefix()
        # assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        # assert self.output_buffer is not None, f"{prefix}output buffer is None; did profile() run?"
        # assert x.device == y.device == self.output_buffer.device, (
        #     f"{prefix}device mismatch: "
        #     f"x={x.device}, y={y.device}, o={self.output_buffer.device}"
        # )

        # self.impl(x, y, self.output_buffer, ctx)
        # return self.output_buffer