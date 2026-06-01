import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.elementwise_impl import elementwise_rr
from ..utils import ElementwiseOpType, Schedule

class Elementwise(vOp):
    """
    Unary elementwise dispatcher for rank-3 logical tensors ``[S, C, D]``.

    This operator dispatches implementation **only based on the input format**
    (``x._format``). The output tensor has the same logical shape as the input.
    Optional scalar parameters ``alpha`` and ``beta`` may be used by certain
    elementwise operations.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, Tuple[Callable, FORMAT]]
        Implementation dispatch table keyed by input format.  
        Each entry maps to ``(callable_impl, resolved_output_format)``.

    alpha : float
        Scalar parameter used by some ops. Default is ``1.0``.

    beta : float
        Scalar parameter used by some ops. Default is ``1.0``.

    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.

    op_type : Optional[ElementwiseOpType]
        The operator type used by the implementation.

    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.

    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer.
    """

    # Implementation dispatch table: keyed only by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: (FORMAT.RAGGED),
        # Add more entries if you support other formats:
        # FORMAT.PAGED: (elementwise_pp, FORMAT.PAGED),
    }

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        assert False, "Elementwise operator is currently disabled pending implementation of the elementwise_rr kernel. Please implement the kernel and update the _impl_map to enable this functionality."
        super().__init__()
        self.impl: Optional[Callable] = None
        self.op_type: Optional[ElementwiseOpType] = None
        self.alpha = alpha
        self.beta = beta
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        """
        Validate input, select the implementation based on ``x._format``,
        allocate the output buffer, and return a ``vTensor`` view.

        Parameters
        ----------
        x : vTensor
            Input tensor. Must be rank-3 with shape ``[S, C, D]``.

        ctx : Context
            Execution context providing runtime ``S`` (``ctx.max_num_pages``)
            and memory tracking.

        Returns
        -------
        vTensor
            A ``vTensor`` view wrapping the allocated output buffer, using the
            resolved output format.

        Raises
        ------
        AssertionError
            If input tensor type, rank, or format is invalid.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, C, D]. Got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[x_fmt]

        # Allocate output buffer on x.device with x.dtype
        S = ctx.max_num_pages             # runtime "S" axis from context
        C, D = x.shape[1], x.shape[2]
        self.output_buffer = torch.empty(
            (S, C, D),
            device=x.device,
            dtype=x.dtype,
        )

        # Account auxiliary memory
        ctx.add_aux_memory(self.output_buffer)

        for t in [x]:
            if t._format == FORMAT.PAGED:
                ctx.add_aux_flops(
                    t.shape[1] * t.shape[2]
                )
        
        # Return vTensor view with dispatched output format
        return as_vtensor(self.output_buffer, self.output_format)

    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Execute the selected implementation into the internal output buffer.

        Expected implementation signature::

            impl(x, output, op_type, alpha, beta, ctx)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor on the same device as the output buffer.

        ctx : Context
            Execution context.

        Returns
        -------
        torch.Tensor
            The output tensor stored in ``self.output_buffer``.

        Raises
        ------
        AssertionError
            If ``profile`` was not called, or device mismatch occurs.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        assert self.output_buffer is not None, f"{prefix}output buffer is None; did profile() run?"
        assert x.device == self.output_buffer.device, (
            f"{prefix}device mismatch: x.device={x.device}, o.device={self.output_buffer.device}"
        )

        self.impl(x, self.output_buffer, self.op_type, self.alpha, self.beta, ctx)
        return self.output_buffer


class Relu(Elementwise):
    r"""
    ReLU-style elementwise operator.

    This operator applies a thresholded linear function:

    .. math::

        \operatorname{out}(x) =
        \begin{cases}
            x, & x \ge \alpha \\
            \beta, & x < \alpha
        \end{cases}

    Parameters
    ----------
    alpha : float, optional
        Threshold value for activation. Default is ``0.0``.

    beta : float, optional
        Value used when :math:`x < \alpha`. Default is ``0.0``.

    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Relu



class Silu(Elementwise):
    r"""
    SiLU-style elementwise operator with affine pre-transform.

    This operator applies:

    .. math::

        \operatorname{SiLU}_{\alpha,\beta}(x)
        = \frac{x}{1 + \exp(\beta x + \alpha)}

    When :math:`\alpha = 0` and :math:`\beta = -1`, this reduces to the
    common SiLU/Swish-like form :math:`x \, \sigma(x)` (up to the chosen
    parameterization in the kernel).

    Parameters
    ----------
    alpha : float, optional
        Bias term inside the exponent, used in :math:`\beta x + \alpha`.
        Default is ``0.0``.

    beta : float, optional
        Scale term inside the exponent, used in :math:`\beta x + \alpha`.
        Default is ``0.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Silu
        

class Sigmoid(Elementwise):
    r"""
    Sigmoid elementwise operator with affine argument.

    This operator applies:

    .. math::

        \sigma_{\alpha,\beta}(x)
        = \frac{1}{1 + \exp(\beta x + \alpha)}

    When :math:`\alpha = 0` and :math:`\beta = -1`, this is the standard
    logistic sigmoid :math:`\sigma(x) = 1 / (1 + e^{-x})`.

    Parameters
    ----------
    alpha : float, optional
        Bias term inside the exponent, used in :math:`\beta x + \alpha`.
        Default is ``0.0``.

    beta : float, optional
        Scale term inside the exponent, used in :math:`\beta x + \alpha`.
        Default is ``0.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Sigmoid
        

class Add_Mul(Elementwise):
    r"""
    Affine elementwise transform.

    This operator applies a simple affine mapping:

    .. math::

        \operatorname{out}(x) = \beta x + \alpha

    With the defaults :math:`\alpha = 0` and :math:`\beta = 1`, this is
    the identity transform :math:`\operatorname{out}(x) = x`.

    Parameters
    ----------
    alpha : float, optional
        Additive bias term :math:`\alpha`. Default is ``0.0``.

    beta : float, optional
        Multiplicative scale term :math:`\beta`. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Add_Mul


class Abs(Elementwise):
    r"""
    Absolute value of an affine transform.

    This operator applies:

    .. math::

        \operatorname{out}(x) = \lvert \beta x + \alpha \rvert

    With the defaults :math:`\alpha = 0` and :math:`\beta = 1`, this
    reduces to the standard absolute value :math:`\lvert x \rvert`.

    Parameters
    ----------
    alpha : float, optional
        Additive bias term inside the absolute value, used in
        :math:`\beta x + \alpha`. Default is ``0.0``.

    beta : float, optional
        Multiplicative scale term inside the absolute value, used in
        :math:`\beta x + \alpha`. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Abs
