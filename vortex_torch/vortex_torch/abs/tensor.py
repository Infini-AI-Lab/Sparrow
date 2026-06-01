from __future__ import annotations
import torch
from enum import Enum
from typing import Any, Iterable
import numbers


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _default_triton_shape(shape: Iterable[int]) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in shape)
    if len(shape) < 2:
        return shape
    return shape[:1] + tuple(_next_power_of_2(dim) for dim in shape[1:])


class FORMAT(Enum):
    """Tensor storage/layout format.

    Attributes:
        BATCHED: Standard dense batched tensors (e.g., ``[B, N, D]``).
        RAGGED: Ragged tensors with variable-length sequences or elements per batch.
        PAGED: Paged tensors used for large or streaming data split into pages/chunks.
    """

    BATCHED = 0
    RAGGED = 1
    PAGED = 2


class vTensor(torch.Tensor):
    """Tensor subclass with `_format` and `tensor_id` metadata fields.

    Rules:
    - Torch ops do NOT change `_format` or `tensor_id`.
    - All vTensors in the same op must have the same `_format`.
    - All vTensors in the same op must have the same `tensor_id`.
    - vTensor CANNOT participate in ops with plain torch.Tensors (raise RuntimeError).
    - vTensor CAN participate in ops with Python scalars (int/float/bool).
    - `tensor_id` must be explicitly provided; no automatic allocation.
    """

    # -------- construction & (de)serialization --------
    def __new__(
        cls,
        data,
        _format: FORMAT = FORMAT.BATCHED,
        tensor_id: int | None = None,
        triton_shape: Iterable[int] | None = None,
        **kwargs,
    ):
        # Accept scalar/iterable/torch.Tensor inputs
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, **kwargs)

        if tensor_id is None:
            tensor_id = -1  # Use -1 as a sentinel for "unassigned" tensor_id

        if not isinstance(tensor_id, int):
            raise TypeError(f"tensor_id must be int, got {type(tensor_id).__name__}")

        # Create subclass without copying storage
        out = torch.Tensor._make_subclass(cls, data, require_grad=data.requires_grad)
        out._format = _format
        out.tensor_id = tensor_id
        out.triton_shape = (
            tuple(int(dim) for dim in triton_shape)
            if triton_shape is not None
            else _default_triton_shape(data.shape)
        )
        return out

    def __repr__(self):
        base = super().__repr__().replace(self.__class__.__name__, "Tensor")
        return (
            f"vTensor(format={self._format}, tensor_id={self.tensor_id}, "
            f"triton_shape={self.triton_shape}, {base})"
        )

    # Pickle support
    def __reduce_ex__(self, protocol):
        return (
            self._rebuild_from_tensor,
            (
                torch.Tensor(self),
                self._format,
                self.tensor_id,
                self.requires_grad,
                self.triton_shape,
            ),
        )

    @staticmethod
    def _rebuild_from_tensor(
        t: torch.Tensor,
        fmt: FORMAT,
        tensor_id: int,
        requires_grad: bool,
        triton_shape: tuple[int, ...] | None = None,
    ):
        out = torch.Tensor._make_subclass(vTensor, t, require_grad=requires_grad)
        out._format = fmt
        out.tensor_id = tensor_id
        out.triton_shape = triton_shape or _default_triton_shape(t.shape)
        return out

    # -------- helpers --------
    @classmethod
    def _from_base(
        cls,
        base: torch.Tensor,
        fmt: FORMAT,
        tensor_id: int,
        triton_shape: Iterable[int] | None = None,
    ):
        if not isinstance(tensor_id, int):
            raise TypeError(f"tensor_id must be int, got {type(tensor_id).__name__}")

        out = torch.Tensor._make_subclass(cls, base, require_grad=base.requires_grad)
        out._format = fmt
        out.tensor_id = tensor_id
        out.triton_shape = (
            tuple(int(dim) for dim in triton_shape)
            if triton_shape is not None
            else _default_triton_shape(base.shape)
        )
        return out

    @staticmethod
    def _iter_tensors(obj: Any) -> Iterable[torch.Tensor]:
        """Recursively yield tensors from nested args/kwargs."""
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                yield from vTensor._iter_tensors(x)
        elif isinstance(obj, dict):
            for x in obj.values():
                yield from vTensor._iter_tensors(x)

    @staticmethod
    def _extract_metadata_and_validate(args, kwargs) -> tuple[FORMAT | None, int | None]:
        """Collect vTensor metadata, ensure no mixing with plain Tensors, and return the op metadata."""
        tensors = list(vTensor._iter_tensors((args, kwargs)))

        # Disallow presence of any plain torch.Tensor alongside vTensor
        has_v = any(isinstance(t, vTensor) for t in tensors)
        has_plain = any(isinstance(t, torch.Tensor) and not isinstance(t, vTensor) for t in tensors)
        if has_v and has_plain:
            raise RuntimeError("vTensor cannot operate with plain torch.Tensor in the same op.")

        # Enforce format consistency
        fmts = {t._format for t in tensors if isinstance(t, vTensor)}
        if len(fmts) > 1:
            raise RuntimeError(f"vTensor _format mismatch in op: {fmts}")

        # Enforce tensor_id consistency
        tensor_ids = {t.tensor_id for t in tensors if isinstance(t, vTensor)}
        if len(tensor_ids) > 1:
            raise RuntimeError(f"vTensor tensor_id mismatch in op: {tensor_ids}")

        fmt = next(iter(fmts)) if fmts else None
        tensor_id = next(iter(tensor_ids)) if tensor_ids else None
        return fmt, tensor_id

    @classmethod
    def _propagate_metadata(cls, result, fmt: FORMAT | None, tensor_id: int | None):
        """Propagate metadata into results (wrap plain Tensors as vTensor)."""
        if fmt is None and tensor_id is None:
            return result  # No vTensor involved => nothing to propagate

        if isinstance(result, torch.Tensor):
            if isinstance(result, vTensor):
                result._format = fmt
                result.tensor_id = tensor_id
                if not hasattr(result, "triton_shape"):
                    result.triton_shape = _default_triton_shape(result.shape)
                return result
            else:
                return cls._from_base(result, fmt, tensor_id)
        elif isinstance(result, (list, tuple)):
            t = type(result)
            return t(cls._propagate_metadata(x, fmt, tensor_id) for x in result)
        elif isinstance(result, dict):
            return {k: cls._propagate_metadata(v, fmt, tensor_id) for k, v in result.items()}
        else:
            # Non-tensor outputs (numbers, bools, None, etc.) are returned as-is
            return result

    @staticmethod
    def _is_allowed_scalar(x: Any) -> bool:
        """Return True if x is a Python numeric/bool scalar allowed to mix with vTensor."""
        return isinstance(x, numbers.Number)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # If no vTensor types are involved, fall back to default behavior.
        if not any(issubclass(t, vTensor) for t in types):
            return super().__torch_function__(func, types, args, kwargs)

        # Validate metadata constraints
        fmt, tensor_id = cls._extract_metadata_and_validate(args, kwargs)

        # Execute op via parent dispatch
        out = super().__torch_function__(func, types, args, kwargs)

        # Propagate metadata into outputs
        return cls._propagate_metadata(out, fmt, tensor_id)


# -------- convenience factory --------
def as_vtensor(
    x: torch.Tensor | Any,
    _format: FORMAT = FORMAT.BATCHED,
    tensor_id: int | None = None,
    triton_shape: Iterable[int] | None = None,
) -> vTensor:
    """Wrap an input as ``vTensor`` without copying storage.

    Args:
        x (torch.Tensor | Any): Input to wrap.
        _format (FORMAT, optional): Desired tensor storage/layout format.
        tensor_id (int | None, optional): Must be explicitly provided.

    Returns:
        vTensor: A ``vTensor`` that references the same underlying storage as ``x``.
    """
    if tensor_id is None:
        tensor_id = -1  # Use -1 as a sentinel for "unassigned" tensor_id

    if not isinstance(tensor_id, int):
        raise TypeError(f"tensor_id must be int, got {type(tensor_id).__name__}")

    if isinstance(x, vTensor):
        x._format = _format
        x.tensor_id = tensor_id
        x.triton_shape = (
            tuple(int(dim) for dim in triton_shape)
            if triton_shape is not None
            else _default_triton_shape(x.shape)
        )
        return x

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    return vTensor._from_base(x, _format, tensor_id, triton_shape=triton_shape)


if __name__ == "__main__":
    shared_id = 1

    a = as_vtensor(torch.randn(2, 3), FORMAT.RAGGED, tensor_id=shared_id)
    b = as_vtensor(torch.ones(2, 3), FORMAT.RAGGED, tensor_id=shared_id)

    # Allowed: vTensor with scalar
    y = a + 2.0
    print("ok scalar:", isinstance(y, vTensor), y._format, y.tensor_id)

    # Allowed: vTensor with vTensor (same format and same tensor_id)
    z = a * b
    print("ok same-id:", isinstance(z, vTensor), z._format, z.tensor_id)

    # Disallowed: vTensor with plain Tensor
    try:
        _ = a + torch.randn(2, 3)
    except RuntimeError as e:
        print("expected error (plain tensor):", e)

    # Disallowed: different formats
    c = as_vtensor(torch.randn(2, 3), FORMAT.PAGED, tensor_id=shared_id)
    try:
        _ = a + c
    except RuntimeError as e:
        print("expected error (format mismatch):", e)

    # Disallowed: different tensor_id
    d = as_vtensor(torch.randn(2, 3), FORMAT.RAGGED, tensor_id=2)
    try:
        _ = a + d
    except RuntimeError as e:
        print("expected error (tensor_id mismatch):", e)

    # Disallowed: missing tensor_id
    try:
        e = as_vtensor(torch.randn(2, 3), FORMAT.RAGGED)
    except ValueError as e:
        print("expected error (missing tensor_id):", e)
