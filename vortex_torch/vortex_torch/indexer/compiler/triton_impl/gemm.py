from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ....abs import FORMAT
def generate_gemm_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    
    input_tensor_id0, input_tensor_id1 = graph.op_to_input_tensor_list[op_id]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    t_i0 = graph.tensor_list[input_tensor_id0]
    t_i1 = graph.tensor_list[input_tensor_id1]
    t_o = graph.tensor_list[output_tensor_id]

    if t_i0._format == FORMAT.BATCHED and t_i1.shape[1] == 1:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.sum((tensor_{input_tensor_id0}_block * tensor_{input_tensor_id1}_block), axis=2)[:,None,:]"
        ]
    else:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.sum((tensor_{input_tensor_id0}_block[:,None,:,:] * tensor_{input_tensor_id1}_block[:,:,None,:]), axis=3)"
        ]

    impl_str = "\n".join(impl_lines)
    return impl_str

