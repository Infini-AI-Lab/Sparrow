from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ...elementwise_binary import Elementwise_Binary
from ....utils import ElementwiseBinaryOpType

def generate_elementwise_binary_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    
    input_tensor_id_0 = graph.op_to_input_tensor_list[op_id][0]
    input_tensor_id_1 = graph.op_to_input_tensor_list[op_id][1]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    t_i0 = graph.tensor_list[input_tensor_id_0]
    t_i1 = graph.tensor_list[input_tensor_id_1]
    t_o = graph.tensor_list[output_tensor_id]
    t_o_shape_str = ", ".join(map(str, t_o.shape))
    assert issubclass(op.__class__, Elementwise_Binary), f"Expected an elementwise binary op, got {graph.op_list[op_id]}"
    alpha = op.alpha
    beta = op.beta

    if op.op_type == ElementwiseBinaryOpType.Add:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = {alpha} * tensor_{input_tensor_id_0}_block + {beta} * tensor_{input_tensor_id_1}_block",
        ]
    elif op.op_type == ElementwiseBinaryOpType.Mul:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tensor_{input_tensor_id_0}_block * tensor_{input_tensor_id_1}_block",
        ]
    elif op.op_type == ElementwiseBinaryOpType.Maximum:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.maximum(tensor_{input_tensor_id_0}_block, tensor_{input_tensor_id_1}_block)",
        ]
    elif op.op_type == ElementwiseBinaryOpType.Minimum:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.minimum(tensor_{input_tensor_id_0}_block, tensor_{input_tensor_id_1}_block)",
        ]
    
    impl_str = "\n".join(impl_lines)
    return impl_str