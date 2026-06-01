import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.save_load_impl import save_rp, load_pr

class Save(vOp):
    r"""
    Format-aware save (copy/convert) dispatcher.

    This operator copies or converts data from an input tensor ``x`` into a
    preallocated output tensor ``o``. Both tensors are treated as rank-3
    objects with logical shape

    .. math::

        [S, D_0, D_1],

    where :math:`S` is a generic leading dimension (it may represent a batch,
    a sequence length, or a page count, depending on the surrounding runtime).

    Key properties:

    - Dispatch is keyed **only** by the format of ``x`` (``x._format``).
    - No internal buffer is allocated; the provided output tensor ``o`` is
      validated and written into directly.
    - The output format is determined by the dispatch table and must match
      ``o._format``.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, Tuple[Callable, FORMAT]]
        Dispatch table keyed by ``x_format``. Each entry maps to
        ``(callable_impl, resolved_output_format)``.

    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.

    output_format : Optional[FORMAT]
        The expected output format for ``o`` as determined in :meth:`profile`.
    """

    # Implementation dispatch table: keyed only by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (save_rp, FORMAT.PAGED),
        # Add more entries if you support other formats.
    }

    def __init__(self):
        assert False, "Save operator is currently disabled pending implementation of the save_rp kernel. Please implement the kernel and update the _impl_map to enable this functionality."
        super().__init__()
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate inputs, resolve the implementation, and return ``o`` as
        the output view.

        This method checks:

        - that both ``x`` and ``o`` are rank-3 ``vTensor`` instances with
          logical shape ``[S, D_0, D_1]``,
        - that their inner dimensions ``D_0`` and ``D_1`` match,
        - that a save implementation is registered for ``x._format``, and
        - that ``o._format`` matches the format required by the dispatch.

        No new buffers are allocated; the provided ``o`` will be used as the
        destination.

        Parameters
        ----------
        x : vTensor
            Input tensor to be copied/converted. Expected logical shape
            ``[S, D_0, D_1]``.

        o : vTensor
            Preallocated output tensor. Must have logical shape compatible
            with ``x`` (matching ``D_0`` and ``D_1``) and the format expected
            by the selected implementation.

        ctx : Context
            Execution context (included for API symmetry; not used for buffer
            allocation in this dispatcher).

        Returns
        -------
        vTensor
            The same object as ``o``, returned as the resolved output view.

        Raises
        ------
        AssertionError
            If types, ranks, shapes, formats, or devices are incompatible,
            or if no implementation is registered for ``x._format``.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"{prefix}profile expects o to be vTensor, got {type(o)}"
        assert x.dim() == 3, f"{prefix}expected 3D x [S, D0, D1], got {tuple(x.shape)}"
        assert o.dim() == 3, f"{prefix}expected 3D o [S, D0, D1], got {tuple(o.shape)}"

        # Shape checks: D0/D1 must match (S may differ by layout; implementation handles it)
        assert x.shape[1] == o.shape[1], (
            f"{prefix}expected matching D0: x.shape[1]={x.shape[1]} vs o.shape[1]={o.shape[1]}"
        )
        assert x.shape[2] == o.shape[2], (
            f"{prefix}expected matching D1: x.shape[2]={x.shape[2]} vs o.shape[2]={o.shape[2]}"
        )

        # Dispatch by x format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[x_fmt]

        # Output format must match the resolved format from dispatch
        assert o._format == self.output_format, (
            f"{prefix}output format mismatch. Expected {self.output_format}, got {o._format}"
        )

        # Device consistency
        assert x.device == o.device, (
            f"{prefix}x and o must be on the same device "
            f"(x.device={x.device}, o.device={o.device})"
        )

        # Save is logically out-of-place, but writes into `o`; return `o` view
        return o

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, o: torch.Tensor, ctx: Context) -> torch.Tensor:
        r"""
        Execute the resolved save operation from ``x`` into ``o``.

        The selected implementation is expected to copy or convert the
        contents of ``x`` into the preallocated tensor ``o`` without
        changing its shape or device.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be copied/converted.

        o : torch.Tensor
            Preallocated output tensor that will receive the data.

        ctx : Context
            Execution context passed through to the implementation.

        Returns
        -------
        torch.Tensor
            The same tensor as ``o``, after the copy/convert operation.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called and no implementation
            is available.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"

        # Expected signature: impl(x, o, ctx)
        self.impl(x, o, ctx)
        return o



class Load(vOp):
    r"""
    Format-aware load (copy/convert) dispatcher.

    This operator copies or converts data from an input tensor ``x`` into an
    internally allocated output buffer. Both input and output are treated as
    rank-3 tensors with logical shape

    .. math::

        [S, D_0, D_1],

    where :math:`S` is a generic leading dimension (it may represent a batch
    size, a sequence length, or a page count, depending on the surrounding
    runtime).

    Key properties
    --------------
    - Dispatch is keyed **only** by the format of ``x`` (``x._format``).
    - The output buffer is allocated during :meth:`profile` using the runtime
      value ``ctx.max_num_pages`` for the leading dimension.
    - The output format is determined by the dispatch table and used when
      wrapping the internal buffer as a :class:`vTensor`.

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
        ``[S_out, D_0, D_1]``, where ``S_out`` comes from the runtime context.
    """

    # Implementation dispatch table: keyed only by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.PAGED: (load_pr, FORMAT.RAGGED),
        # Add more entries if you support other formats.
    }

    def __init__(self):
        assert False, "Load operator is currently disabled pending implementation of the load_pr kernel. Please implement the kernel and update the _impl_map to enable this functionality."
        super().__init__()
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate the input, select an implementation, allocate the output
        buffer, and return an :func:`as_vtensor` view with the resolved format.

        The input tensor is expected to have logical shape ``[S_in, D_0, D_1]``.
        The output tensor is allocated with shape

        .. math::

            [S_{\text{out}}, D_0, D_1],

        where :math:`S_{\text{out}}` is taken from ``ctx.max_num_pages`` to
        match the runtime configuration for the leading dimension.

        Parameters
        ----------
        x : vTensor
            Input tensor to be loaded from, with logical shape
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
        assert x.dim() == 3, f"{prefix}expected 3D x [S, D0, D1], got {tuple(x.shape)}"

        # Dispatch by x format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[x_fmt]

        # Allocate output buffer [S_out, D0, D1].
        # S_out comes from runtime context (e.g., number of tokens/pages).
        S_out = ctx.max_num_pages
        D0, D1 = x.shape[1], x.shape[2]
        self.output_buffer = torch.empty(
            (S_out, D0, D1),
            device=x.device,
            dtype=x.dtype,
        )
        ctx.add_aux_memory(self.output_buffer)

        # Return vTensor view with dispatched output format
        return as_vtensor(self.output_buffer, self.output_format)

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        r"""
        Run the selected load operation into the internally allocated buffer.

        The selected implementation is expected to copy or convert the
        contents of ``x`` into the internal buffer stored in
        :attr:`output_buffer`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be copied/converted, on the same device as the
            internal output buffer.

        ctx : Context
            Execution context passed through to the implementation.

        Returns
        -------
        torch.Tensor
            The internally allocated output tensor stored in
            :attr:`output_buffer`.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called (no implementation or
            internal output buffer available).
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        assert self.output_buffer is not None, (
            f"{prefix}internal output buffer is None; did profile() run?"
        )

        # Expected signature: impl(x, output, ctx)
        self.impl(x, self.output_buffer, ctx)
        return self.output_buffer
