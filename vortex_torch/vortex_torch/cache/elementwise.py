import torch
from ..abs import vOp
from .context import Context
from .triton_kernels.elementwise_impl import elementwise_pp, elementwise_rp, elementwise_pr, elementwise_rr
from ..abs import vTensor, FORMAT, as_vtensor
from ..utils import ElementwiseOpType
from typing import Tuple, Dict, Callable, Optional

class Elementwise(vOp):
    r"""
    Unary elementwise operator dispatcher (e.g. ReLU/Sigmoid/SiLU/Abs/Affine).

    This class dispatches a family of unary elementwise operations on
    rank-3 tensors. The input is treated as

    .. math::

        X \in \mathbb{R}^{B \times N \times D},

    where:

    - :math:`B` is a leading batch-like axis (for example,
      ``max_new_tokens_per_batch * head_num`` coming from the runtime
      context),
    - :math:`N` is a sequence or position dimension, and
    - :math:`D` is a feature/channel dimension.

    The operation is applied pointwise:

    .. math::

        Y[b, n, d] = f(X[b, n, d]; \alpha, \beta, \text{op_type}),

    where the actual function :math:`f` is selected by :attr:`op_type`,
    and may make use of scalar parameters :attr:`alpha` and :attr:`beta`
    (for example, in affine or activation variants).

    Dispatch is based on the pair of tensor formats
    ``(x_format, o_format)`` and a registry mapping:

    .. code-block:: text

        (x_format, o_format) -> (impl, resolved_output_format)

    Policy
    ------
    - If ``output`` is ``None``, :meth:`profile` selects an implementation
      with ``o_format == FORMAT.RAGGED``, allocates an internal buffer
      of logical shape ``[B, N, D]``, and returns a ``vTensor`` view.
    - If ``output`` is provided, :meth:`profile` requires an exact
      ``(x_fmt, o_fmt)`` mapping in :attr:`_impl_map` and validates
      shape/device consistency.
    - The logical (``N, D``) axes are preserved by design; only the
      leading ``B`` comes from the runtime context.

    Attributes
    ----------
    _impl_map : Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]]
        Dispatch table keyed by ``(x_format, o_format)``. Each entry maps
        to ``(callable_impl, resolved_output_format)``.
    alpha : float
        Scalar parameter used by certain unary ops.
    beta : float
        Scalar parameter used by certain unary ops.
    op_type : Optional[ElementwiseOpType]
        Runtime-set enum/int describing the specific elementwise operation.
    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    output_buffer : Optional[torch.Tensor]
        Internal output buffer allocated when ``output`` is ``None``.
    """

    # Implementation registry:
    #   key   = (x_format, o_format)
    #   value = (callable_impl, resolved_output_format)
    _impl_map: Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.PAGED,  FORMAT.PAGED):  (elementwise_pp, FORMAT.PAGED),
        (FORMAT.PAGED,  FORMAT.RAGGED): (elementwise_pr, FORMAT.RAGGED),
        (FORMAT.RAGGED, FORMAT.PAGED):  (elementwise_rp, FORMAT.PAGED),
        (FORMAT.RAGGED, FORMAT.RAGGED): (elementwise_rr, FORMAT.RAGGED),
    }

    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        r"""
        Initialize a unary elementwise dispatcher.

        Parameters
        ----------
        alpha : float, optional
            Scalar parameter used by certain elementwise operations
            (for example, as an additive or bias term). Default is ``0.0``.

        beta : float, optional
            Scalar parameter used by certain elementwise operations
            (for example, as a multiplicative or slope term).
            Default is ``1.0``.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.op_type: Optional[ElementwiseOpType] = None          # runtime-set enum/int for the op
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    # ------------------------------ helpers ------------------------------ #
    def _infer_impl_ragged(self, x_fmt: FORMAT) -> Tuple[Callable, FORMAT]:
        r"""
        Infer an implementation assuming a RAGGED output format.

        This helper is used when :meth:`profile` is called with
        ``output is None``. It selects an implementation for the
        key ``(x_fmt, FORMAT.RAGGED)`` in :attr:`_impl_map`.

        Parameters
        ----------
        x_fmt : FORMAT
            The format of the input tensor ``x``.

        Returns
        -------
        (Callable, FORMAT)
            The implementation callable and the resolved output format.

        Raises
        ------
        AssertionError
            If there is no entry for ``(x_fmt, FORMAT.RAGGED)`` in
            :attr:`_impl_map`.
        """
        key = (x_fmt, FORMAT.RAGGED)
        assert key in self._impl_map, (
            f"{self._prefix()}no RAGGED-output implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        return self._impl_map[key]  # -> (impl, out_fmt)

    # --------------------------------------------------------------------- #
    # profile: validate, select impl/format, and optionally allocate output
    # --------------------------------------------------------------------- #
    def profile(
        self, x: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""
        Validate inputs, select implementation and output format, and
        optionally allocate an internal output buffer.

        The input tensor ``x`` is expected to have logical shape
        ``[B, N, D]``. The auxiliary tensor ``loc`` carries per-position
        metadata used by the implementation (for example, mapping positions
        to segments or other runtime indices); its exact shape and semantics
        are defined by the kernel.

        There are two modes:

        - **No output provided** (``output is None``):

          - Select an implementation for ``(x._format, FORMAT.RAGGED)``.
          - Allocate an internal buffer with shape
            ``[B, N, D]``, where

            .. math::

                B = \text{ctx.max_new_tokens_per_batch} \times \text{ctx.head_num},

          - Wrap it as a :class:`vTensor` with the resolved output format.

        - **Output provided** (``output is not None``):

          - Require an exact mapping for ``(x._format, output._format)``.
          - Validate that ``output`` has rank 3 and preserves the
            ``(N, D)`` dimensions of ``x``.
          - Validate device consistency between ``x`` and ``output``.

        Parameters
        ----------
        x : vTensor
            Input tensor with logical shape ``[B, N, D]``.

        output : Optional[vTensor]
            Optional preallocated output tensor. If ``None``, an internal
            buffer is allocated; otherwise, this tensor must have shape
            ``[B_out, N, D]`` for some ``B_out`` and a format compatible
            with :attr:`_impl_map`.

        loc : torch.Tensor
            Auxiliary tensor carrying per-position metadata required by
            the implementation (e.g., location/segment indices).

        ctx : Context
            Execution context that provides the runtime value of ``B``
            (via ``ctx.max_new_tokens_per_batch`` and ``ctx.head_num``)
            and is used for auxiliary memory accounting.

        Returns
        -------
        vTensor
            A :class:`vTensor` view representing the resolved output:
            either the provided ``output`` or an internally allocated
            buffer.

        Raises
        ------
        AssertionError
            If types, ranks, formats, shapes, or devices are incompatible,
            or if no implementation is found in :attr:`_impl_map`.
        """
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"

        x_fmt = x._format
        N, D = x.shape[1], x.shape[2]

        # Case A: output not provided -> choose RAGGED impl and allocate buffer
        if output is None:
            self.impl, self.output_format = self._infer_impl_ragged(x_fmt)

            # Allocate on x.device/x.dtype; B comes from runtime context
            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = torch.empty(
                (B, N, D),
                device=x.device,
                dtype=x.dtype,
            )
            ctx.add_aux_memory(self.output_buffer)
            return as_vtensor(self.output_buffer, self.output_format)

        # Case B: output provided -> validate and select exact impl
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, "
            f"got ndim={output.dim()} shape={tuple(output.shape)}"
        )

        o_fmt = output._format
        key = (x_fmt, o_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, o_fmt={o_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[key]

        # Shape consistency: unary elementwise keeps (N,D)
        assert output.shape[1] == N and output.shape[2] == D, (
            f"{prefix}output shape mismatch. Expected (*,{N},{D}), got {tuple(output.shape)}"
        )

        # Device consistency check
        assert x.device == output.device, (
            f"{prefix}x and output must be on the same device "
            f"(x.device={x.device}, output.device={output.device})"
        )

        return output

    # --------------------------------------------------------------------- #
    # execute: run selected impl and return the plain output tensor
    # --------------------------------------------------------------------- #
    def execute(
        self, x: torch.Tensor, output: Optional[torch.Tensor], loc: torch.Tensor, ctx: Context
    ) -> torch.Tensor:
        r"""
        Execute the selected unary elementwise implementation.

        This method assumes that :meth:`profile` has already selected an
        implementation and, if needed, allocated an internal output buffer.

        Parameters
        ----------
        x : torch.Tensor
            Plain input tensor with shape consistent with the ``vTensor``
            validated in :meth:`profile`.

        output : Optional[torch.Tensor]
            Optional preallocated output tensor. If ``None``, the internal
            buffer created during :meth:`profile` will be used.

        loc : torch.Tensor
            Auxiliary tensor carrying per-position metadata required by the
            implementation (e.g., location/segment indices).

        ctx : Context
            Execution context forwarded to the implementation.

        Returns
        -------
        torch.Tensor
            The output tensor written by the implementation: either the
            provided ``output`` or the internal buffer.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called and no implementation
            or internal buffer is available.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}called before profile() (impl is None)"

        if output is None:
            assert self.output_buffer is not None, (
                f"{prefix}internal output buffer is None; did profile() run?"
            )
            output = self.output_buffer

        # Expected signature for impl:
        #   impl(x, output, loc, ctx, op_type, alpha, beta)
        self.impl(x, output, loc, ctx, self.op_type, self.alpha, self.beta)
        return output



class Relu(Elementwise):
    r"""
    Piecewise ReLU-like activation.

    This operator applies, elementwise, the scalar function

    .. math::

        f(x; \alpha, \beta) =
        \begin{cases}
            x,      & x \ge \alpha, \\
            \beta,  & x < \alpha.
        \end{cases}

    Given an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is defined by

    .. math::

        Y[b, n, d] = f\bigl(X[b, n, d]; \alpha, \beta\bigr).

    Parameters
    ----------
    alpha : float, optional
        Threshold value :math:`\alpha`. Inputs greater than or equal to
        this threshold are passed through unchanged. Default is ``0.0``.

    beta : float, optional
        Fallback value :math:`\beta` used when :math:`x < \alpha`.
        Default is ``0.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Relu


class Silu(Elementwise):
    r"""
    SiLU-like activation with configurable shift and slope.

    This operator applies, elementwise, the scalar function

    .. math::

        \operatorname{SiLU}(x; \alpha, \beta)
        = \frac{x}{1 + \exp(\beta x + \alpha)}.

    Given an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is

    .. math::

        Y[b, n, d]
        = \operatorname{SiLU}\bigl(X[b, n, d]; \alpha, \beta\bigr).

    Parameters
    ----------
    alpha : float, optional
        Bias term :math:`\alpha` added inside the exponential. Default is
        ``0.0``.

    beta : float, optional
        Slope :math:`\beta` multiplying :math:`x` inside the exponential.
        Default is ``0.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Silu


class Sigmoid(Elementwise):
    r"""
    Sigmoid activation with configurable shift and slope.

    This operator applies, elementwise, the scalar function

    .. math::

        \sigma(x; \alpha, \beta)
        = \frac{1}{1 + \exp(\beta x + \alpha)}.

    Given an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is

    .. math::

        Y[b, n, d]
        = \sigma\bigl(X[b, n, d]; \alpha, \beta\bigr).

    Parameters
    ----------
    alpha : float, optional
        Bias term :math:`\alpha` added inside the exponential. Default is
        ``0.0``.

    beta : float, optional
        Slope :math:`\beta` multiplying :math:`x` inside the exponential.
        Default is ``0.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Sigmoid


class Add_Mul(Elementwise):
    r"""
    Affine transformation :math:`y = \beta x + \alpha`.

    This operator applies, elementwise, the scalar function

    .. math::

        f(x; \alpha, \beta) = \beta x + \alpha.

    For an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is

    .. math::

        Y[b, n, d]
        = \beta \, X[b, n, d] + \alpha.

    Parameters
    ----------
    alpha : float, optional
        Additive term :math:`\alpha` in the affine transform. Default is
        ``0.0``.

    beta : float, optional
        Multiplicative term :math:`\beta` in the affine transform.
        Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Add_Mul


class Abs(Elementwise):
    r"""
    Absolute-value transform of an affine argument.

    This operator applies, elementwise, the scalar function

    .. math::

        f(x; \alpha, \beta) = \bigl|\beta x + \alpha\bigr|.

    For an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is

    .. math::

        Y[b, n, d]
        = \bigl|\beta \, X[b, n, d] + \alpha\bigr|.

    Parameters
    ----------
    alpha : float, optional
        Additive term :math:`\alpha` inside the absolute value. Default is
        ``0.0``.

    beta : float, optional
        Multiplicative term :math:`\beta` inside the absolute value.
        Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Abs
