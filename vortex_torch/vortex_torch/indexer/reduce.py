import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.reduce_impl import reduce_rr
from ..utils import ReduceType, Schedule

class Reduce(vOp):
    r"""
    Generic reduction dispatcher for rank-3 logical tensors ``[N, D_0, D_1]``.

    This operator performs a 1D reduction over either the ``D_0`` or ``D_1``
    axis of a 3D tensor. The leading dimension ``N`` is generic and may
    represent a batch axis (``B``) or a sequence/page axis (``S``); the
    reduction is applied independently for each of the ``N`` slices.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    the output logical shape depends on the configured reduction dimension
    ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out} \in \mathbb{R}^{N \times 1 \times D_1}.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out} \in \mathbb{R}^{N \times D_0 \times 1}.

    The specific reduction operation (e.g. mean, max, min, L2-norm, sum)
    is selected via :attr:`reduce_type`.

    Dispatch is keyed only by the input format ``x._format``.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, Tuple[Callable, FORMAT]]
        Dispatch table keyed by ``x_format``. Each entry maps to
        ``(callable_impl, resolved_output_format)``.

    dim : int
        Reduction dimension in the logical 3D tensor: must be either

        - ``1`` for reduction over the :math:`D_0` axis, or
        - ``2`` for reduction over the :math:`D_1` axis.

    reduce_type : Optional[ReduceType]
        The type of reduction to perform (e.g. mean, max, min, L2-norm, sum).

    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.

    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.

    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer with logical shape
        ``[N, out_D0, out_D1]``, where ``out_D0`` and ``out_D1`` depend on
        ``dim`` as described above.
    """

    # Implementation dispatch table: keyed only by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: (FORMAT.RAGGED),
        FORMAT.BATCHED: (FORMAT.BATCHED),
        # Add more entries if you support other formats, e.g.:
        # FORMAT.PAGED: (reduce_pp, FORMAT.PAGED),
    }

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.reduce_type: Optional[ReduceType] = None
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W
        # Validate reduction dimension at construction
        prefix = self._prefix()
        assert self.dim in (1, 2), f"{prefix}__init__: dim must be 1 or 2, got dim={self.dim}"

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate the input, select an implementation based on ``x._format``,
        allocate the output buffer, and return a ``vTensor`` view.

        The input tensor is expected to have logical shape ``[N, D_0, D_1]``,
        where the leading dimension ``N`` may represent either a batch size or
        a sequence/page count. The runtime uses ``ctx.max_num_pages`` to define
        the leading dimension of the output, in line with other operators that
        treat the first axis as the logical ``N`` axis.

        According to :attr:`dim`, the output logical shape is:

        - ``dim == 1`` → ``[N, 1, D_1]``
        - ``dim == 2`` → ``[N, D_0, 1]``

        Parameters
        ----------
        x : vTensor
            Input tensor with logical shape ``[N, D_0, D_1]``.

        ctx : Context
            Execution context providing ``ctx.max_num_pages`` for the leading
            dimension and tracking auxiliary memory usage.

        Returns
        -------
        vTensor
            A ``vTensor`` view wrapping the allocated output buffer with the
            resolved output format.

        Raises
        ------
        AssertionError
            If ``x`` is not a :class:`vTensor`, if its rank is not 3, or if no
            implementation is registered for ``x._format``.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [N, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        # Compute output logical shape according to `dim`
        # The leading dimension N is taken from the runtime context,
        # not from x.shape[0], to remain consistent with other ops.
        D0, D1 = x.shape[1], x.shape[2]
        out_D0 = 1 if self.dim == 1 else D0   # D_0 collapsed when reducing over dim=1
        out_D1 = 1 if self.dim == 2 else D1   # D_1 collapsed when reducing over dim=2

        # Allocate output buffer on x.device with x.dtype
        self.output_buffer = as_vtensor(torch.empty(
            (0, out_D0, out_D1),
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
        
        # Return vTensor view carrying the dispatched output format
        return self.output_buffer

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        r"""
        Run the selected reduction implementation into the internal buffer
        and return the result.

        The underlying implementation is expected to follow the signature::

            impl(x, output, dim, reduce_type, ctx)

        where ``dim`` specifies which logical axis to reduce and
        :attr:`reduce_type` selects the reduction operation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``[N, D_0, D_1]`` on the same device as
            the preallocated output buffer.

        ctx : Context
            Execution context passed through to the implementation.

        Returns
        -------
        torch.Tensor
            The output tensor stored in ``self.output_buffer``, with logical
            shape determined by :attr:`dim` as described in :meth:`profile`.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called (no implementation or
            output buffer).
        """

        assert False
        # prefix = self._prefix()
        # assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        # assert self.output_buffer is not None, f"{prefix}output buffer is None; did profile() run?"

        # # Expected signature: impl(x, output, dim, reduce_type, ctx)
        # self.impl(x, self.output_buffer, self.dim, self.reduce_type, ctx)
        # return self.output_buffer


    
class Max(Reduce):
    r"""
    Maximum reduction over a single logical axis.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    this operator computes, depending on ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out}[n, 0, d_1]
         = \max_{0 \le d_0 < D_0} X[n, d_0, d_1],

      with shape :math:`[N, 1, D_1]`.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out}[n, d_0, 0]
         = \max_{0 \le d_1 < D_1} X[n, d_0, d_1],

      with shape :math:`[N, D_0, 1]`.

    The leading dimension :math:`N` may represent either a batch axis
    (``B``) or a sequence/page axis (``S``); the reduction is applied
    independently for each slice along this dimension.

    Parameters
    ----------
    dim : int, optional
        Reduction dimension in the logical 3D tensor (``1`` for :math:`D_0`,
        ``2`` for :math:`D_1`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Max


class Min(Reduce):
    r"""
    Minimum reduction over a single logical axis.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    this operator computes, depending on ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out}[n, 0, d_1]
         = \min_{0 \le d_0 < D_0} X[n, d_0, d_1],

      with shape :math:`[N, 1, D_1]`.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out}[n, d_0, 0]
         = \min_{0 \le d_1 < D_1} X[n, d_0, d_1],

      with shape :math:`[N, D_0, 1]`.

    The leading dimension :math:`N` may represent either a batch axis
    (``B``) or a sequence/page axis (``S``); the reduction is applied
    independently for each slice along this dimension.

    Parameters
    ----------
    dim : int, optional
        Reduction dimension in the logical 3D tensor (``1`` for :math:`D_0`,
        ``2`` for :math:`D_1`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Min


class Mean(Reduce):
    r"""
    Mean reduction over a single logical axis.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    this operator computes, depending on ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out}[n, 0, d_1]
         = \frac{1}{D_0} \sum_{d_0=0}^{D_0-1} X[n, d_0, d_1],

      with shape :math:`[N, 1, D_1]`.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out}[n, d_0, 0]
         = \frac{1}{D_1} \sum_{d_1=0}^{D_1-1} X[n, d_0, d_1],

      with shape :math:`[N, D_0, 1]`.

    The leading dimension :math:`N` may represent either a batch axis
    (``B``) or a sequence/page axis (``S``); the reduction is applied
    independently for each slice along this dimension.

    Parameters
    ----------
    dim : int, optional
        Reduction dimension in the logical 3D tensor (``1`` for :math:`D_0`,
        ``2`` for :math:`D_1`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Mean


class L2Norm(Reduce):
    r"""
    L2-norm reduction over a single logical axis.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    this operator computes, depending on ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out}[n, 0, d_1]
         = \sqrt{\sum_{d_0=0}^{D_0-1} X[n, d_0, d_1]^2},

      with shape :math:`[N, 1, D_1]`.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out}[n, d_0, 0]
         = \sqrt{\sum_{d_1=0}^{D_1-1} X[n, d_0, d_1]^2},

      with shape :math:`[N, D_0, 1]`.

    The leading dimension :math:`N` may represent either a batch axis
    (``B``) or a sequence/page axis (``S``); the reduction is applied
    independently for each slice along this dimension.

    Parameters
    ----------
    dim : int, optional
        Reduction dimension in the logical 3D tensor (``1`` for :math:`D_0`,
        ``2`` for :math:`D_1`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.L2Norm


class Sum(Reduce):
    r"""
    Sum reduction over a single logical axis.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    this operator computes, depending on ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out}[n, 0, d_1]
         = \sum_{d_0=0}^{D_0-1} X[n, d_0, d_1],

      with shape :math:`[N, 1, D_1]`.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out}[n, d_0, 0]
         = \sum_{d_1=0}^{D_1-1} X[n, d_0, d_1],

      with shape :math:`[N, D_0, 1]`.

    The leading dimension :math:`N` may represent either a batch axis
    (``B``) or a sequence/page axis (``S``); the reduction is applied
    independently for each slice along this dimension.

    Parameters
    ----------
    dim : int, optional
        Reduction dimension in the logical 3D tensor (``1`` for :math:`D_0`,
        ``2`` for :math:`D_1`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Sum
