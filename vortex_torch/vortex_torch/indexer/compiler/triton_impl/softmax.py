from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ....utils import INDENT
from ....abs import FORMAT
from ...scan import Softmax


def _triton_shape(t) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(t, "triton_shape", tuple(t.shape)))


def _and_masks(*parts: str) -> str:
    parts = [part for part in parts if part]
    return " & ".join(parts)


def generate_softmax_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    x_d0, x_d1 = int(t_i.shape[-2]), int(t_i.shape[-1])
    x_d0_triton, x_d1_triton = _triton_shape(t_i)[-2:]
    softmax_mask = _and_masks(
        "p_mask[:, None, None]",
        "(d0_idx[None, :, None] < x_D0)" if x_d0 != x_d0_triton else "",
        "(d1_idx[None, None, :] < x_D1)" if x_d1 != x_d1_triton else "",
    )
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Softmax), f"Expected a softmax op, got {graph.op_list[op_id]}"
    assert t_i._format == FORMAT.RAGGED, f"Expected ragged input tensor for softmax, got {t_i._format}"
    assert t_o._format == FORMAT.RAGGED, f"Expected ragged output tensor for softmax, got {t_o._format}"
    
    func_def_lines = [
    f"@triton.jit",
    f"def softmax_kernel(",
    f"{INDENT}x,",
    f"{INDENT}out,",
    f"{INDENT}indptr,",
    f"{INDENT}scale,",
    f"{INDENT}bos: tl.constexpr,",
    f"{INDENT}eos: tl.constexpr,",
    f"{INDENT}topk_val: tl.constexpr,",
    f"{INDENT}x_D0: tl.constexpr,",
    f"{INDENT}x_D1: tl.constexpr,",
    f"{INDENT}x_D0_TRITON: tl.constexpr,",
    f"{INDENT}x_D1_TRITON: tl.constexpr,",
    f"{INDENT}BLOCK_P: tl.constexpr = 512,",
    f"):",
    f"{INDENT}pid = tl.program_id(0)",
    f"",
    f"{INDENT}start = tl.load(indptr + pid)",
    f"{INDENT}end = tl.load(indptr + pid + 1)",
    f"{INDENT}num_pages_this_seq = end - start",
    f"",
    f"{INDENT}threshold: tl.constexpr = bos + eos + topk_val",
    f"{INDENT}if num_pages_this_seq <= threshold:",
    f"{INDENT*2}return",
    f"",
    f"{INDENT}num_pages_to_compute = num_pages_this_seq - bos - eos",
    f"{INDENT}if num_pages_to_compute <= 0:",
    f"{INDENT*2}return",
    f"",
    f"{INDENT}page_stride = x_D0 * x_D1",
    f"",
    f"{INDENT}x_base_ptr = x + (start + bos) * page_stride",
    f"{INDENT}out_base_ptr = out + (start + bos) * page_stride",
    f"",
    f"{INDENT}d0_idx = tl.arange(0, x_D0_TRITON)",
    f"{INDENT}d1_idx = tl.arange(0, x_D1_TRITON)",
    f"{INDENT}p_idx = tl.arange(0, BLOCK_P)",
    f"",
    f"{INDENT}neg_inf = -1e30",
    f"{INDENT}m = tl.full((x_D0_TRITON, x_D1_TRITON), neg_inf, dtype=tl.float32)",
    f"{INDENT}s = tl.zeros((x_D0_TRITON, x_D1_TRITON), dtype=tl.float32)",
    f"",
    f"{INDENT}for p in range(0, num_pages_to_compute, BLOCK_P):",
    f"{INDENT*2}kp = tl.minimum(BLOCK_P, num_pages_to_compute - p)",
    f"{INDENT*2}p_mask = p_idx < kp",
    f"",
    f"{INDENT*2}offs = (",
    f"{INDENT*3}(p + p_idx)[:, None, None] * page_stride",
    f"{INDENT*3}+ d0_idx[None, :, None] * x_D1",
    f"{INDENT*3}+ d1_idx[None, None, :]",
    f"{INDENT*2}).to(tl.int32)",
    f"",
    f"{INDENT*2}mask = {softmax_mask}",
    f"{INDENT*2}slab = tl.load(x_base_ptr + offs, mask=mask, other=neg_inf).to(tl.float32)",
    f"{INDENT*2}slab = slab * scale",
    f"",
    f"{INDENT*2}mc = tl.max(slab, axis=0)",
    f"{INDENT*2}sc = tl.sum(tl.exp(slab - mc[None, :, :]), axis=0)",
    f"",
    f"{INDENT*2}m_new = tl.maximum(m, mc)",
    f"{INDENT*2}s = s * tl.exp(m - m_new) + sc * tl.exp(mc - m_new)",
    f"{INDENT*2}m = m_new",
    f"",
    f"{INDENT}for p in range(0, num_pages_to_compute, BLOCK_P):",
    f"{INDENT*2}kp = tl.minimum(BLOCK_P, num_pages_to_compute - p)",
    f"{INDENT*2}p_mask = p_idx < kp",
    f"",
    f"{INDENT*2}offs = (",
    f"{INDENT*3}(p + p_idx)[:, None, None] * page_stride",
    f"{INDENT*3}+ d0_idx[None, :, None] * x_D1",
    f"{INDENT*3}+ d1_idx[None, None, :]",
    f"{INDENT*2}).to(tl.int32)",
    f"",
    f"{INDENT*2}mask = {softmax_mask}",
    f"{INDENT*2}slab = tl.load(x_base_ptr + offs, mask=mask, other=neg_inf).to(tl.float32)",
    f"{INDENT*2}slab = slab * scale",
    f"{INDENT*2}slab = tl.exp(slab - m[None, :, :]) / s[None, :, :]",
    f"{INDENT*2}slab = slab.to(tl.bfloat16)",
    f"{INDENT*2}tl.store(out_base_ptr + offs, slab, mask=mask)",
]   
    ctx.auxilary_func_def_lines.extend(func_def_lines)

    impl_lines = [
    f"{INDENT}eff_batch_size = ctx.batch_size * ctx.num_kv_heads",
    f"",
    f"{INDENT}softmax_kernel[(eff_batch_size,)](",
    f"{INDENT*2}tensor_{input_tensor_id},",
    f"{INDENT*2}tensor_{output_tensor_id},",
    f"{INDENT*2}ctx.dense_kv_indptr,",
    f"{INDENT*2}{op.scale},",
    f"{INDENT*2}ctx.block_reserved_bos,",
    f"{INDENT*2}ctx.block_reserved_eos,",
    f"{INDENT*2}ctx.topk_val,",
    f"{INDENT*2}{x_d0},",
    f"{INDENT*2}{x_d1},",
    f"{INDENT*2}{x_d0_triton},",
    f"{INDENT*2}{x_d1_triton},",
    f"{INDENT*2}num_warps=4,",
    f"{INDENT*2}num_stages=1,",
    f"{INDENT})",
    ]

    return "\n".join(impl_lines)
