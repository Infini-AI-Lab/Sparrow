
AVAILABLE_IMPL_BACKENDS = {}

from .triton_impl import generate_triton_impl
AVAILABLE_IMPL_BACKENDS["triton"] = generate_triton_impl

