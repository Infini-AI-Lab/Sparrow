import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp, as_vtensor
from .triton_kernels.softmax_impl import softmax_inplace_r
from .triton_kernels.normalize_impl import normalize_inplace_r
from ..utils import Schedule

class Softmax(vOp):
    r"""
    In-place softmax dispatcher over a packed leading axis.

    The input is treated as a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{S_{\text{pack}} \times D_0 \times D_1},

    where the leading dimension :math:`S_{\text{pack}}` is a packed
    concatenation of :math:`B` segments:

    .. math::

        S_{\text{pack}}
        = \sum_{b=0}^{B-1} S_b, \qquad
        X =
        \begin{bmatrix}
            X_0 \\
            X_1 \\
            \vdots \\
            X_{B-1}
        \end{bmatrix},

    with

    .. math::

        X_b \in \mathbb{R}^{S_b \times D_0 \times D_1}.
    
    For each segment :math:`b` and each fixed pair
    :math:`(d_0, d_1)`, this operator applies a scaled softmax along the
    packed axis within that segment:

    .. math::

        \text{out}[s, d_0, d_1]
        =
        \frac{\exp\bigl(\text{scale} \cdot X[s, d_0, d_1]\bigr)}
             {\sum_{s' \in \mathcal{I}_b}
                \exp\bigl(\text{scale} \cdot X[s', d_0, d_1]\bigr)},
        \quad s \in \mathcal{I}_b,

    where :math:`\mathcal{I}_b` denotes the index range in
    :math:`[0, S_{\text{pack}})` corresponding to the :math:`b`-th
    segment of length :math:`S_b`.

    In other words, ``dim == 0`` is a **packed S axis**, and softmax is
    applied independently within each segment of that axis.

    Key properties
    --------------
    - Only ``dim == 0`` is supported.
    - Dispatch is keyed by the input tensor format ``x._format``.
    - No output buffer is allocated; the operation is performed in-place
      on ``x`` and the same tensor is returned.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, Tuple[Callable, FORMAT]]
        Dispatch table keyed by ``x_format``. Each entry maps to
        ``(callable_impl, resolved_output_format)``.
    dim : int
        Softmax axis. Must be ``0`` and corresponds to the packed
        :math:`S_{\text{pack}}` dimension.
    scale : float
        Multiplicative factor applied to ``x`` before the softmax
        (i.e. computes softmax of ``x * scale``).
    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    """

    # Implementation dispatch table: keyed by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: (FORMAT.RAGGED),
        # Extend with other formats if you add more kernels, e.g.:
        # FORMAT.PAGED: (softmax_inplace_p, FORMAT.PAGED),
    }

    def __init__(self, dim: int = 0, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.schedule = Schedule.S
        # Validate dim at construction
        prefix = self._prefix()
        assert self.dim in (0,), f"{prefix}__init__: dim must be 0, got dim={self.dim}"

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate the input and select an implementation.

        Since the operation is in-place, no output buffer is allocated and
        this method simply returns the input :class:`vTensor` unmodified.

        The input is expected to have logical shape
        ``[S_pack, D_0, D_1]``, where ``S_pack`` is understood as a packed
        concatenation of :math:`B` segments of lengths :math:`S_b`. The
        softmax is applied along ``dim == 0`` **within** each such segment.

        Parameters
        ----------
        x : vTensor
            Input tensor to be modified in-place, with logical shape
            ``[S_pack, D_0, D_1]``.

        ctx : Context
            Execution context (included for API symmetry; not used for
            buffer allocation here).

        Returns
        -------
        vTensor
            The same object as ``x``, returned as the output view.

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
            f"{prefix}expected 3D input [S_pack, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        self.output_buffer = as_vtensor(torch.empty(
            (0, x.shape[1], x.shape[2]),
            device=x.device,
            dtype=x.dtype,
        ), self.output_format, tensor_id=len(ctx.tensor_list)  # Assign a new tensor_id based on current tensor count
        )
        # Track auxiliary memory and graph structure in the context
        ctx.tensor_list.append(self.output_buffer)  # Track the output buffer in the context
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # Map the output tensor to this operation
        ctx.op_list.append(self)  # Track this operation in the context
        ctx.op_to_input_tensor_list.append([x.tensor_id])  # Map this op to its input tensors
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # Map this op to its output tensor

        return self.output_buffer

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        r"""
        Execute the in-place scaled softmax and return the input tensor.

        Conceptually, this computes a segment-wise softmax of
        ``x * scale`` along the packed axis ``dim == 0``, where the
        segments along that axis correspond to
        :math:`S_0, S_1, \dots, S_{B-1}` and are treated independently.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be modified in-place. Must be compatible with
            the shape and format validated during :meth:`profile`.

        ctx : Context
            Execution context passed through to the implementation.

        Returns
        -------
        torch.Tensor
            The same tensor instance ``x``, after the in-place softmax.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called and no implementation is
            available.
        """
        assert False, "Softmax execution is currently disabled pending implementation of the softmax_inplace_r kernel. Please implement the kernel and update the _impl_map to enable this functionality."
        # prefix = self._prefix()
        # assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"

        # # Expected signature: impl(x, dim, scale, ctx)
        # self.impl(x, self.dim, self.scale, ctx)
        # return x


    

class Normalize(vOp):
    r"""
    In-place normalization dispatcher over a packed leading axis.

    The input is treated as a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{S_{\text{pack}} \times D_0 \times D_1},

    where the leading dimension :math:`S_{\text{pack}}` is a packed
    concatenation of :math:`B` segments:

    .. math::

        S_{\text{pack}}
        = \sum_{b=0}^{B-1} S_b, \qquad
        X =
        \begin{bmatrix}
            X_0 \\
            X_1 \\
            \vdots \\
            X_{B-1}
        \end{bmatrix},

    with

    .. math::

        X_b \in \mathbb{R}^{S_b \times D_0 \times D_1}.
    
    For each segment :math:`b` and each fixed pair :math:`(d_0, d_1)`,
    this operator normalizes the values along the packed axis within that
    segment. A common interpretation is L2 normalization:

    .. math::

        \hat{X}[s, d_0, d_1]
        =
        \frac{X[s, d_0, d_1]}
             {\sqrt{\sum_{s' \in \mathcal{I}_b}
                     X[s', d_0, d_1]^2}},
        \quad s \in \mathcal{I}_b,

    where :math:`\mathcal{I}_b` denotes the index range in
    :math:`[0, S_{\text{pack}})` corresponding to the :math:`b`-th
    segment of length :math:`S_b`.

    In other words, ``dim == 0`` is a **packed S axis**, and normalization
    is applied independently within each segment of that axis.

    Key properties
    --------------
    - Only ``dim == 0`` is supported.
    - Dispatch is keyed by the input tensor format ``x._format``.
    - No output buffer is allocated; the operation is performed in-place
      on ``x`` and the same tensor is returned.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, Tuple[Callable, FORMAT]]
        Dispatch table keyed by ``x_format``. Each entry maps to
        ``(callable_impl, resolved_output_format)``.
    dim : int
        Normalization axis. Must be ``0`` and corresponds to the packed
        :math:`S_{\text{pack}}` dimension.
    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    """

    # Implementation dispatch table: keyed by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: (FORMAT.RAGGED),
        # Extend with other formats if you add more kernels, e.g.:
        # FORMAT.PAGED: (normalize_inplace_p, FORMAT.PAGED),
    }

    def __init__(self, dim: int = 0):
        assert False, "Normalize operator is currently disabled pending implementation of the normalize_inplace_r kernel. Please implement the kernel and update the _impl_map to enable this functionality."
        super().__init__()
        self.dim = dim
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None

        # Validate dim at construction
        prefix = self._prefix()
        assert self.dim in (0,), f"{prefix}__init__: dim must be 0, got dim={self.dim}"

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate the input and select an implementation.

        Since the operation is in-place, no output buffer is allocated and
        this method simply returns the input :class:`vTensor` unmodified.

        The input is expected to have logical shape
        ``[S_pack, D_0, D_1]``, where ``S_pack`` is understood as a packed
        concatenation of :math:`B` segments of lengths :math:`S_b`. The
        normalization is applied along ``dim == 0`` **within** each such
        segment.

        Parameters
        ----------
        x : vTensor
            Input tensor to be normalized in-place, with logical shape
            ``[S_pack, D_0, D_1]``.

        ctx : Context
            Execution context (included for API symmetry; not used for
            buffer allocation here).

        Returns
        -------
        vTensor
            The same object as ``x``, returned as the output view.

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
            f"{prefix}expected 3D input [S_pack, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        self.output_buffer = as_vtensor(torch.empty(
            (0, x.shape[1], x.shape[2]),
            device=x.device,
            dtype=x.dtype,
        ), self.output_format, tensor_id=len(ctx.tensor_list)  # Assign a new tensor_id based on current tensor count
        )
        # Track auxiliary memory and graph structure in the context
        ctx.tensor_list.append(self.output_buffer)  # Track the output buffer in the context
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # Map the output tensor to this operation
        ctx.op_list.append(self)  # Track this operation in the context
        ctx.op_to_input_tensor_list.append([x.tensor_id])  # Map this op to its input tensors
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # Map this op to its output tensor

        return self.output_buffer

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        r"""
        Execute the in-place normalization and return the input tensor.

        Conceptually, this performs segment-wise normalization along the
        packed axis ``dim == 0``, where the segments along that axis
        correspond to :math:`S_0, S_1, \dots, S_{B-1}` and are treated
        independently.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be normalized in-place. Must be compatible with
            the shape and format validated during :meth:`profile`.

        ctx : Context
            Execution context passed through to the implementation.

        Returns
        -------
        torch.Tensor
            The same tensor instance ``x``, after in-place normalization.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called and no implementation is
            available.
        """
        assert False, "Normalize execution is currently disabled pending implementation of the normalize_inplace_r kernel. Please implement the kernel and update the _impl_map to enable this functionality."
        # prefix = self._prefix()
        # assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"

        # # Expected signature: impl(x, dim, ctx)
        # self.impl(x, self.dim, ctx)
        # return x
