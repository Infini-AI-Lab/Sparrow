import torch
from ..abs import vOp
from .context import Context
from .triton_kernels.fill_impl import fill_p
from ..abs import vTensor, FORMAT
from typing import Dict, Callable, Optional

class Fill(vOp):
    r"""
    In-place page-wise fill dispatcher.

    This operator performs an in-place, format-aware fill over a batched
    tensor. The input is treated as a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times D_0 \times D_1},

    where:

    - :math:`B` is a batch-like axis (for example, batch * heads),
    - :math:`D_0, D_1` encode per-page/per-token features.

    Using per-position metadata stored in ``loc`` together with layout
    information in :class:`Context`, the implementation identifies
    positions where a page has reached its logical end (e.g. an
    end-of-page or end-of-sequence condition) and fills the corresponding
    tiles with a scalar value :attr:`alpha`:

    .. math::

        X[b, d_0, d_1] \leftarrow \alpha
        \quad \text{for all positions } (b, d_0, d_1)
        \text{ that lie past the end-of-page mark.}

    All modifications are performed in-place; the tensor format
    (PAGED, RAGGED, etc.) determines how page boundaries are interpreted
    and which elements are affected.

    Key properties
    --------------
    - Dispatch is keyed only by the input tensor format ``x._format``.
    - The operator is purely in-place: no output buffer is allocated.
    - The same :class:`vTensor` is returned from :meth:`profile`.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, Callable]
        Dispatch table keyed by ``x_format``. Each entry maps to a
        callable implementation that performs the in-place fill.

    alpha : float
        Scalar fill value written into selected positions.

    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.
    """

    # Implementation registry keyed by x_format.
    # Expected impl signature: impl(x: torch.Tensor, loc: torch.Tensor, ctx: Context, alpha: float) -> None
    _impl_map: Dict[FORMAT, Callable] = {
        FORMAT.PAGED: fill_p,  # e.g., wraps `fill_p_kernel` (page-major in-place fill)
        # Add more entries if you support other layouts, e.g.:
        # FORMAT.RAGGED: fill_r,
    }

    def __init__(self, alpha: float = 0.0):
        r"""
        Initialize an in-place fill operator.

        Parameters
        ----------
        alpha : float, optional
            Scalar fill value used to overwrite positions after the
            end-of-page condition is met. Default is ``0.0``.
        """
        super().__init__()
        self.alpha = alpha
        self.impl: Optional[Callable] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, loc: torch.Tensor, ctx: Context) -> vTensor:
        r"""
        Validate inputs and select the in-place implementation.

        Since the operation is in-place, no output buffer is allocated and
        this method simply returns the input :class:`vTensor` unmodified.

        Parameters
        ----------
        x : vTensor
            Input tensor to be modified in-place. Expected logical shape
            ``[B, D_0, D_1]``.

        loc : torch.Tensor
            Auxiliary tensor carrying page/position metadata used to
            detect end-of-page locations.

        ctx : Context
            Execution context providing layout information and any
            additional metadata needed by the implementation.

        Returns
        -------
        vTensor
            The same object as ``x``, returned as the output view.

        Raises
        ------
        AssertionError
            If ``x`` is not a :class:`vTensor`, if its rank is not 3, if
            ``loc`` is not a :class:`torch.Tensor`, or if no
            implementation is registered for ``x._format``.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), (
            f"{prefix}profile expects loc to be torch.Tensor, got {type(loc)}"
        )
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [B, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl = self._impl_map[x_fmt]

        # In-place: return the same vTensor view
        return x

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, loc: torch.Tensor, ctx: Context) -> torch.Tensor:
        r"""
        Execute the in-place page-wise fill and return the input tensor.

        The selected implementation examines page/position metadata in
        ``loc`` and context, and overwrites elements in ``x`` with the
        scalar value :attr:`alpha` once an end-of-page condition is
        detected.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be modified in-place. Must be compatible with
            the :class:`vTensor` validated in :meth:`profile`.

        loc : torch.Tensor
            Auxiliary tensor carrying page/position metadata.

        ctx : Context
            Execution context forwarded to the implementation.

        Returns
        -------
        torch.Tensor
            The same tensor instance ``x``, after in-place filling.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called and no implementation
            is available.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"

        # Expected signature: impl(x, loc, ctx, alpha)
        self.impl(x, loc, ctx, self.alpha)
        return x