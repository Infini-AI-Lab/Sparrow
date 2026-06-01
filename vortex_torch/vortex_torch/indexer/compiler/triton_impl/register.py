from ...matmul import GeMM
from ...reduce import Reduce
from ...elementwise_binary import Elementwise_Binary
from ...output_func import topK
from ...scan import Softmax

from .gemm import generate_gemm_impl
from .reduce import generate_reduce_impl
from .elementwise_binary import generate_elementwise_binary_impl
from .topk import generate_topk_impl
from .softmax import generate_softmax_impl
IMPL_REGISTRY = {
    GeMM: generate_gemm_impl,
    Reduce: generate_reduce_impl,
    Elementwise_Binary: generate_elementwise_binary_impl,
    topK: generate_topk_impl,
    Softmax: generate_softmax_impl,
}

def get_impl_func(op) -> str:
    for op_type, impl_func in IMPL_REGISTRY.items():
        if issubclass(op.__class__, op_type):
            return impl_func
    raise NotImplementedError(f"No implementation function found for op type {op.__class__}")