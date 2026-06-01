from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ...reduce import Reduce
from ....utils import ReduceType


def _has_padded_inner_dim(t) -> bool:
    triton_shape = tuple(int(dim) for dim in getattr(t, "triton_shape", tuple(t.shape)))
    return int(t.shape[1]) != triton_shape[1] or int(t.shape[2]) != triton_shape[2]


def generate_reduce_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    t_o_shape_str = ", ".join(map(str, t_o.shape[1:]))
    input_block = f"tensor_{input_tensor_id}_block"
    if input_tensor_id in graph.input_tensor_ids and _has_padded_inner_dim(t_i):
        input_mask = f"tensor_{input_tensor_id}_mask"
        if getattr(t_i._format, "name", None) == "BATCHED":
            input_mask = f"{input_mask}[None, :, :]"
    else:
        input_mask = None
    assert issubclass(op.__class__, Reduce), f"Expected a reduce op, got {graph.op_list[op_id]}"
    if op.reduce_type == ReduceType.Sum:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.sum(tensor_{input_tensor_id}_block, keep_dims=True, axis={op.dim})",
        ]
    elif op.reduce_type == ReduceType.Max:
        if input_mask is not None:
            input_block = f"tl.where({input_mask}, {input_block}, -3.402823e38)"
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.max({input_block}, keep_dims=True, axis={op.dim})",
        ]
    elif op.reduce_type == ReduceType.Min:
        if input_mask is not None:
            input_block = f"tl.where({input_mask}, {input_block}, 3.402823e38)"
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.min({input_block}, keep_dims=True, axis={op.dim})",
        ]
    elif op.reduce_type == ReduceType.L2Norm:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.sqrt(tl.sum((tensor_{input_tensor_id}_block * tensor_{input_tensor_id}_block).to(tl.float32), keep_dims=True, axis={op.dim}))",
        ]
    elif op.reduce_type == ReduceType.Mean:
        compute_line = f"tensor_{output_tensor_id}_block = tl.sum(tensor_{input_tensor_id}_block, keep_dims=True, axis={op.dim}) * ({1.0 / t_i.shape[op.dim]})"
        if input_tensor_id in graph.input_tensor_ids and getattr(t_i._format, "name", None) == "BATCHED":
            impl_lines = [
                f"if tensor_{input_tensor_id}_cache_miss:",
                f"    {compute_line}",
            ]
        else:
            impl_lines = [compute_line]
    impl_str = "\n".join(impl_lines)
    return impl_str
