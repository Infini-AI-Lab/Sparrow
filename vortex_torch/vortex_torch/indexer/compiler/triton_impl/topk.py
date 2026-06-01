from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ....abs import FORMAT
from ....utils import INDENT
def generate_topk_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    ctx.compilation_header_lines.append("from vortex_torch_C import topk_output, topk_output_v2")  # Import the custom C++ extension for topk
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    assert t_i._format == FORMAT.RAGGED, f"Expected ragged input tensor for topk, got {t_i._format}"
    assert t_o._format == FORMAT.RAGGED, f"Expected ragged output tensor for topk, got {t_o._format}"
    
    impl_lines = [
        f"topk_output(",
        f"{INDENT * 2}tensor_{input_tensor_id},",
        f"{INDENT * 2}ctx.dense_kv_indptr,",
        f"{INDENT * 2}ctx.sparse_kv_indptr,",
        f"{INDENT * 2}ctx.dense_kv_indices,",
        f"{INDENT * 2}tensor_{output_tensor_id},",
        f"{INDENT * 2}ctx.batch_size * ctx.num_kv_heads,",
        f"{INDENT * 2}ctx.block_reserved_bos,",
        f"{INDENT * 2}ctx.block_reserved_eos,",
        f"{INDENT * 2}ctx.max_num_blocks_per_request,",
        f")",
    ]
    impl_str = "\n".join(impl_lines)
    return impl_str
