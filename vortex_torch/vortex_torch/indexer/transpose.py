import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.transpose_impl import transpose_rr

class Transpose(vOp):
    r"""
    Transpose dispatcher for rank-3 logical tensors.

    This operator transposes the last two dimensions of a rank-3 tensor
    while keeping the leading axis unchanged. The input is treated as

    .. math::

        X \in \mathbb{R}^{S \times D_0 \times D_1},

    and the output has logical shape

    .. math::

        Y \in \mathbb{R}^{S \times D_1 \times D_0},

    with

    .. math::

        Y[s, d_1, d_0] = X[s, d_0, d_1].

    The leading dimension :math:`S` may represent a true sequence axis or
    a packed axis (e.g. :math:`S_{\text{pack}} = \sum_b S_b`); the transpose
    is applied independently for each slice along that axis.

    Dispatch is keyed only by the input tensor format ``x._format``.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, Tuple[Callable, FORMAT]]
        Dispatch table keyed by ``x_format``. Each entry maps to
        ``(callable_impl, resolved_output_format)``.
    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer with logical shape
        ``[S, D_1, D_0]``.
    """

    # Implementation dispatch table: keyed only by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (transpose_rr, FORMAT.RAGGED),
        # Add more entries if you support other formats, e.g.:
        # FORMAT.PAGED: (transpose_pp, FORMAT.PAGED),
    }

    def __init__(self):
        assert False, "Transpose operator is currently disabled pending implementation of the transpose_rr kernel. Please implement the kernel and update the _impl_map to enable this functionality."
        super().__init__()
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate the input, select an implementation, allocate the output
        buffer, and return a :class:`vTensor` view with the resolved format.

        The input tensor is expected to have logical shape
        ``[S_in, D_0, D_1]``. The output buffer is allocated with shape

        .. math::

            [S, D_1, D_0],

        where :math:`S` is taken from ``ctx.max_num_pages`` to match the
        runtime configuration for the leading dimension.

        Parameters
        ----------
        x : vTensor
            Input tensor to be transposed, with logical shape
            ``[S_in, D_0, D_1]``.

        ctx : Context
            Execution context providing ``ctx.max_num_pages`` for the leading
            dimension and tracking auxiliary memory usage.

        Returns
        -------
        vTensor
            A ``vTensor`` view wrapping the internally allocated output
            buffer with the resolved output format.

        Raises
        ------
        AssertionError
            If ``x`` is not a :class:`vTensor`, if its rank is not 3, or if
            no implementation is registered for ``x._format``.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[x_fmt]

        # Allocate output buffer: [S, D1, D0]
        # S is derived from runtime context (number of pages/tokens in the pipeline)
        S = ctx.max_num_pages
        D0, D1 = x.shape[1], x.shape[2]
        self.output_buffer = torch.empty(
            (S, D1, D0),
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

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        r"""
        Run the selected transpose implementation and return the output buffer.

        The implementation transposes the last two dimensions of ``x`` into
        the internal buffer stored in :attr:`output_buffer`, leaving the
        leading dimension unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be transposed, on the same device as the
            internal output buffer.

        ctx : Context
            Execution context passed through to the implementation.

        Returns
        -------
        torch.Tensor
            The internally allocated output tensor with shape
            ``[S, D_1, D_0]``.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called and the internal output
            buffer or implementation is not available.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        assert self.output_buffer is not None, (
            f"{prefix}internal output buffer is None; did profile() run?"
        )

        # Expected signature: impl(x, output, ctx)
        self.impl(x, self.output_buffer, ctx)
        return self.output_buffer
