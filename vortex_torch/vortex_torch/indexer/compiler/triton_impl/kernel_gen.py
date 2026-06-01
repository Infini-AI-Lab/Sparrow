from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ....abs import FORMAT
from ....utils import Schedule, INDENT
from .register import get_impl_func
import os


def _logical_shape(t) -> tuple[int, ...]:
    return tuple(int(dim) for dim in t.shape)


def _triton_shape(t) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(t, "triton_shape", tuple(t.shape)))


def _inner_shapes(t) -> tuple[int, int, int, int]:
    logical = _logical_shape(t)
    triton = _triton_shape(t)
    return logical[1], logical[2], triton[1], triton[2]


def _and_masks(*parts: str) -> str | None:
    parts = [part for part in parts if part]
    if not parts:
        return None
    return " & ".join(parts)


def indent_block(text: str, level: int = 1) -> str:
    """
    Indent a multi-line text block by the given indentation level.
    """
    prefix = INDENT * level
    lines = text.splitlines()
    return "\n".join(prefix + line if line.strip() else line for line in lines)


def generate_initialization_str(sub_graph: Graph, ctx:Context) -> str:
    """
    Generate the initialization code string for a given sub-graph.
    This may include declarations of intermediate variables and pre-computations.
    """
    lines = []

    for local_tensor_id in sub_graph.input_tensor_ids:
        t = sub_graph.tensor_list[local_tensor_id]
        _, _, t1, t2 = _inner_shapes(t)
        if t._format in (FORMAT.BATCHED, FORMAT.PAGED):
            lines.append(f"# Declare variables for tensor_{local_tensor_id}")
            lines.append(f"tensor_{local_tensor_id}_dim1_ptr = tl.arange(0, {t1})")
            lines.append(f"tensor_{local_tensor_id}_dim2_ptr = tl.arange(0, {t2})")
        if t._format == FORMAT.BATCHED:
            lines.append(
                f"tensor_{local_tensor_id}_block = tl.zeros((1, {t1}, {t2}), dtype=tl.float32)"
            )
            lines.append(f"tensor_{local_tensor_id}_cached_batch_idx = tl.full((), -1, dtype=tl.int32)")

    for local_tensor_id, t in enumerate(sub_graph.tensor_list):
        if local_tensor_id in sub_graph.input_tensor_ids:
            continue
        if t._format == FORMAT.BATCHED:
            _, _, t1, t2 = _inner_shapes(t)
            lines.append(
                f"tensor_{local_tensor_id}_block = tl.zeros((1, {t1}, {t2}), dtype=tl.float32)"
            )

    
    for local_tensor_id in sub_graph.output_tensor_ids:
        t = sub_graph.tensor_list[local_tensor_id]
        _, _, t1, t2 = _inner_shapes(t)
        if t._format in (FORMAT.RAGGED, FORMAT.PAGED):
            lines.append(f"tensor_{local_tensor_id}_dim1_ptr = tl.arange(0, {t1})")
            lines.append(f"tensor_{local_tensor_id}_dim2_ptr = tl.arange(0, {t2})")

    if ctx.num_pages_per_workload > 1:
        lines.append(f"page_idx_i32_ptr = tl.arange(0, {ctx.num_pages_per_workload})")
        lines.append(f"block_i32_ptr = tl.arange(0, {ctx.num_blocks_per_page})")
                
    return "\n".join(lines) if lines else "# No initialization required"


def generate_load_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    """
    Generate the tensor loading code string for a given sub-graph.
    """
    batched_lines = []
    paged_lines = []
    ragged_lines = []
    for local_tensor_id in sub_graph.input_tensor_ids:
        t = sub_graph.tensor_list[local_tensor_id]
        l1, l2, t1, t2 = _inner_shapes(t)
        if t._format == FORMAT.BATCHED:
            batched_lines.append(f"tensor_{local_tensor_id}_dim1_ptr = tl.arange(0, {t1})")
            batched_lines.append(f"tensor_{local_tensor_id}_dim2_ptr = tl.arange(0, {t2})")
            batched_lines.append(f"tensor_{local_tensor_id}_cache_miss = new_batch_idx_i32 != tensor_{local_tensor_id}_cached_batch_idx")
            batched_lines.append(f"if tensor_{local_tensor_id}_cache_miss:")
            batched_lines.append(f"{INDENT}tensor_{local_tensor_id}_block_ptr = tensor_{local_tensor_id}_ptr + new_batch_idx_i32 * {l1 * l2} + tensor_{local_tensor_id}_dim1_ptr[:, None] * {l2} + tensor_{local_tensor_id}_dim2_ptr[None, :]")
            dim_mask = _and_masks(
                f"(tensor_{local_tensor_id}_dim1_ptr[:, None] < {l1})" if l1 != t1 else "",
                f"(tensor_{local_tensor_id}_dim2_ptr[None, :] < {l2})" if l2 != t2 else "",
            )
            if dim_mask is None:
                batched_lines.append(f"{INDENT}tensor_{local_tensor_id}_block = tl.load(tensor_{local_tensor_id}_block_ptr, cache_modifier=\".ca\").to(tl.float32)[None, :, :]")
            else:
                batched_lines.append(f"{INDENT}tensor_{local_tensor_id}_mask = {dim_mask}")
                batched_lines.append(f"{INDENT}tensor_{local_tensor_id}_block = tl.load(tensor_{local_tensor_id}_block_ptr, mask=tensor_{local_tensor_id}_mask, other=0.0, cache_modifier=\".ca\").to(tl.float32)[None, :, :]")
            batched_lines.append(f"{INDENT}tensor_{local_tensor_id}_cached_batch_idx = new_batch_idx_i32")
        elif t._format == FORMAT.PAGED:

            if ctx.num_pages_per_workload == 1:
                paged_lines.append(f"tensor_{local_tensor_id}_block_ptr = tensor_{local_tensor_id}_ptr + (page_idx_i32 + workload_ptr)[:, None, None] * {l1 * l2} + tensor_{local_tensor_id}_dim1_ptr[None, :, None] * {l2} + tensor_{local_tensor_id}_dim2_ptr[None, None, :]")
                dim_mask = _and_masks(
                    f"(tensor_{local_tensor_id}_dim1_ptr[None, :, None] < {l1})" if l1 != t1 else "",
                    f"(tensor_{local_tensor_id}_dim2_ptr[None, None, :] < {l2})" if l2 != t2 else "",
                )
                if dim_mask is None:
                    paged_lines.append(f"tensor_{local_tensor_id}_block = tl.load(tensor_{local_tensor_id}_block_ptr, cache_modifier=\".cv\").to(tl.float32)")
                else:
                    paged_lines.append(f"tensor_{local_tensor_id}_mask = {dim_mask}")
                    paged_lines.append(f"tensor_{local_tensor_id}_block = tl.load(tensor_{local_tensor_id}_block_ptr, mask=tensor_{local_tensor_id}_mask, other=0.0, cache_modifier=\".cv\").to(tl.float32)")
            else:
                paged_lines.append(f"tensor_{local_tensor_id}_block_ptr = tensor_{local_tensor_id}_ptr + page_indices_i32[:,None,None,None] * {l1 * l2} + block_i32_ptr[None,:,None,None] * {l1 * l2} + tensor_{local_tensor_id}_dim1_ptr[None,None,:,None] * {l2} + tensor_{local_tensor_id}_dim2_ptr[None,None,None,:]")
                paged_lines.append(f"tensor_{local_tensor_id}_mask = {_and_masks('page_valid[:, None, None, None]', f'(tensor_{local_tensor_id}_dim1_ptr[None,None,:,None] < {l1})' if l1 != t1 else '', f'(tensor_{local_tensor_id}_dim2_ptr[None,None,None,:] < {l2})' if l2 != t2 else '')}")
                paged_lines.append(f"tensor_{local_tensor_id}_block = tl.reshape(tl.load(tensor_{local_tensor_id}_block_ptr, mask=tensor_{local_tensor_id}_mask, other=0.0, cache_modifier=\".cv\"), ({ctx.workload_chunk_size}, {t1}, {t2})).to(tl.float32)")
            
        elif t._format == FORMAT.RAGGED:
            ragged_lines.append(f"tensor_{local_tensor_id}_dim1_ptr = tl.arange(0, {t1})")
            ragged_lines.append(f"tensor_{local_tensor_id}_dim2_ptr = tl.arange(0, {t2})")
            ragged_lines.append(f"tensor_{local_tensor_id}_block_ptr = tensor_{local_tensor_id}_ptr + (ragged_idx_i32 + workload_ptr)[:, None, None] * {l1 * l2} + tensor_{local_tensor_id}_dim1_ptr[None, :, None] * {l2} + tensor_{local_tensor_id}_dim2_ptr[None, None, :]")
            dim_mask = _and_masks(
                f"(tensor_{local_tensor_id}_dim1_ptr[None, :, None] < {l1})" if l1 != t1 else "",
                f"(tensor_{local_tensor_id}_dim2_ptr[None, None, :] < {l2})" if l2 != t2 else "",
            )
            if dim_mask is None:
                ragged_lines.append(f"tensor_{local_tensor_id}_block = tl.load(tensor_{local_tensor_id}_block_ptr).to(tl.float32)")
            else:
                ragged_lines.append(f"tensor_{local_tensor_id}_mask = {dim_mask}")
                ragged_lines.append(f"tensor_{local_tensor_id}_block = tl.load(tensor_{local_tensor_id}_block_ptr, mask=tensor_{local_tensor_id}_mask, other=0.0).to(tl.float32)")

    if batched_lines:
        batched_lines = [
            "new_batch_idx_i32 = tl.load(winfo_x_indices + i).to(tl.int32)",
            indent_block("\n".join(batched_lines), 0),
        ]
        batched_lines = "\n".join(batched_lines)

    paged_lines = "\n".join(paged_lines) if paged_lines else "# No paged tensor loading required"
    ragged_lines = "\n".join(ragged_lines) if ragged_lines else "# No ragged tensor loading required"
    addressing_lines = [
        "ragged_idx_i32 = tl.load(winfo_y_offsets + i).to(tl.int32)", 
        "page_idx_i32 = tl.load(indices + ragged_idx_i32).to(tl.int32)" if ctx.num_pages_per_workload == 1 else f"page_indices_i32 = tl.load(indices + ragged_idx_i32 + page_idx_i32_ptr * {ctx.num_blocks_per_page}, mask=page_valid, other=0).to(tl.int32)",
    ]
    addressing_lines = "\n".join(addressing_lines)

    return "\n\n".join(filter(None, [batched_lines, addressing_lines, paged_lines, ragged_lines])) if (batched_lines or addressing_lines or paged_lines or ragged_lines) else "# No tensor loading required"


def generate_store_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    """
    Generate the tensor storing code string for a given sub-graph.
    """
    paged_lines = []
    ragged_lines = []
    for local_tensor_id in sub_graph.output_tensor_ids:
        t = sub_graph.tensor_list[local_tensor_id]
        l1, l2, t1, t2 = _inner_shapes(t)
        if t._format == FORMAT.PAGED:
            if ctx.num_pages_per_workload == 1:
                paged_lines.append(f"tensor_{local_tensor_id}_block_ptr = tensor_{local_tensor_id}_ptr + (page_idx_i32 + workload_ptr)[:, None, None] * {l1 * l2} + tensor_{local_tensor_id}_dim1_ptr[None, :, None] * {l2} + tensor_{local_tensor_id}_dim2_ptr[None, None, :]")
                dim_mask = _and_masks(
                    f"(tensor_{local_tensor_id}_dim1_ptr[None, :, None] < {l1})" if l1 != t1 else "",
                    f"(tensor_{local_tensor_id}_dim2_ptr[None, None, :] < {l2})" if l2 != t2 else "",
                )
                if dim_mask is None:
                    paged_lines.append(f"tl.store(tensor_{local_tensor_id}_block_ptr, tensor_{local_tensor_id}_block.to(tl.bfloat16))")
                else:
                    paged_lines.append(f"tensor_{local_tensor_id}_mask = {dim_mask}")
                    paged_lines.append(f"tl.store(tensor_{local_tensor_id}_block_ptr, tensor_{local_tensor_id}_block.to(tl.bfloat16), mask=tensor_{local_tensor_id}_mask)")
            else:
                paged_lines.append(f"tensor_{local_tensor_id}_block_ptr = tensor_{local_tensor_id}_ptr + page_indices_i32[:,None,None,None] * {l1 * l2} + block_i32_ptr[None,:,None,None] * {l1 * l2} + tensor_{local_tensor_id}_dim1_ptr[None,None,:,None] * {l2} + tensor_{local_tensor_id}_dim2_ptr[None,None,None,:]")
                paged_lines.append(f"tensor_{local_tensor_id}_mask = {_and_masks('page_valid[:, None, None, None]', f'(tensor_{local_tensor_id}_dim1_ptr[None,None,:,None] < {l1})' if l1 != t1 else '', f'(tensor_{local_tensor_id}_dim2_ptr[None,None,None,:] < {l2})' if l2 != t2 else '')}")
                paged_lines.append(f"tl.store(tensor_{local_tensor_id}_block_ptr, tl.reshape(tensor_{local_tensor_id}_block, ({ctx.num_pages_per_workload}, {ctx.num_blocks_per_page}, {t1}, {t2})).to(tl.bfloat16), mask=tensor_{local_tensor_id}_mask)")

        elif t._format == FORMAT.RAGGED:
            ragged_lines.append(
                "\n".join([
                    f"tensor_{local_tensor_id}_block_ptr = tensor_{local_tensor_id}_ptr + ragged_idx_i32 * {l1 * l2} + workload_ptr[:,None,None] * {l1 * l2} + tensor_{local_tensor_id}_dim1_ptr[None,:,None] * {l2} + tensor_{local_tensor_id}_dim2_ptr[None,None,:]",
                ])
            )
            
            ragged_lines.append(f"tensor_{local_tensor_id}_mask = {_and_masks('valid[:, None, None]', f'(tensor_{local_tensor_id}_dim1_ptr[None,:,None] < {l1})' if l1 != t1 else '', f'(tensor_{local_tensor_id}_dim2_ptr[None,None,:] < {l2})' if l2 != t2 else '')}")
            ragged_lines.append(f"tl.store(tensor_{local_tensor_id}_block_ptr, tensor_{local_tensor_id}_block.to(tl.bfloat16), mask=tensor_{local_tensor_id}_mask)")

    
    if paged_lines:
        paged_lines = "\n".join(paged_lines)
    
    if ragged_lines:
        ragged_lines = [
            f"_len = tl.load(winfo_y_lens + i)",
            f"valid = workload_ptr < _len",
            "\n".join(ragged_lines)
        ] if ctx.num_pages_per_workload == 1 else ragged_lines
        ragged_lines = "\n".join(ragged_lines)
    return "\n\n".join(filter(None, [paged_lines, ragged_lines])) if (paged_lines or ragged_lines) else "# No tensor loading required"


def generate_computation_str(sub_graph: Graph, ctx: Context) -> str:
    
    lines = []
    for op_id, op in enumerate(sub_graph.op_list):
        op_impl_func = get_impl_func(op)
        impl_str = op_impl_func(sub_graph, op_id, ctx)
        lines.append(impl_str)
    return "\n\n".join(lines) if lines else "# No computation required"

def generate_triton_kernel(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context
) -> str:
    """
    Generate a Triton kernel definition for a given sub-graph.
    """
    kernel_arg_list = [
        "indices,                # int32",
        "winfo_x_indices,        # int32",
        "winfo_y_offsets,        # int32",
        "winfo_y_lens,           # int32",
        "winfo_num_workloads,    # int32",
    ]

    # Collect input tensor kernel arguments
    for local_tensor_id in sub_graph.input_tensor_ids:
        tensor_name = f"tensor_{local_tensor_id}"
        kernel_arg_list.extend([
            f"{tensor_name}_ptr,",
            f"{tensor_name}_dim0: tl.constexpr,",
        ])

    # Collect output tensor kernel arguments
    for local_tensor_id in sub_graph.output_tensor_ids:
        tensor_name = f"tensor_{local_tensor_id}"
        kernel_arg_list.extend([
            f"{tensor_name}_ptr,",
            f"{tensor_name}_dim0: tl.constexpr,",
        ])

    kernel_args = "\n".join(f"{INDENT}{arg}" for arg in kernel_arg_list)
    prepare_workload_lines = []
    if ctx.num_pages_per_workload > 1:
        prepare_workload_lines = [
        f"_len = tl.load(winfo_y_lens + i)",
        f"valid = workload_ptr < _len",
        f"_page = (_len + {ctx.num_blocks_per_page - 1}) // {ctx.num_blocks_per_page}",
        f"page_valid = page_idx_i32_ptr < _page",
     ]
    prepare_workload_str = "\n".join(prepare_workload_lines) if prepare_workload_lines else "# No workload preparation required"

    initialization_str = indent_block(generate_initialization_str(sub_graph,ctx), 1)
    prepare_workload_str = indent_block(prepare_workload_str, 2)
    load_tensor_str = indent_block(generate_load_tensor_str(sub_graph, ctx), 2)
    store_tensor_str = indent_block(generate_store_tensor_str(sub_graph, ctx), 2)
    computation_str = indent_block(generate_computation_str(sub_graph, ctx), 2)
    kernel_str = f"""
@triton.jit
def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_kernel(
{kernel_args}
):
    # ------------------------------------------------------------
    # Program-level partitioning of workloads
    # ------------------------------------------------------------
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    per = n_workloads // num_progs
    r = n_workloads % num_progs
    start = pid * per + tl.minimum(pid, r)
    end = start + per + (pid < r)
    workload_ptr = tl.arange(0, {ctx.workload_chunk_size})

{initialization_str}

    for i in range(start, end):
{prepare_workload_str}
{load_tensor_str}
{computation_str}
{store_tensor_str}
"""
    return kernel_str.strip()


def generate_triton_impl(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context
) -> str:
    """
    Generate a Triton kernel and its Python wrapper implementation.
    """
    ctx.compilation_header_lines.extend([
        "import torch",
        "import triton",
        "import triton.language as tl",
    ])
    if sub_graph.schedule == Schedule.W:
        kernel_str = generate_triton_kernel(sub_graph, sub_graph_id, ctx)

        arg_list = []
        kernel_input_list = [
            "ctx.dense_kv_indices",
            "ctx.winfo_q_indices",
            "ctx.winfo_kv_offsets",
            "ctx.winfo_kv_lens",
            "ctx.winfo_num_workloads",
        ]

        # Collect input tensor arguments
        for local_tensor_id in sub_graph.input_tensor_ids:
            tensor_name = f"tensor_{local_tensor_id}"
            arg_list.append(tensor_name)
            kernel_input_list.extend([
                tensor_name,
                f"{tensor_name}.shape[0]",
            ])

        # Collect output tensor arguments
        for local_tensor_id in sub_graph.output_tensor_ids:
            tensor_name = f"tensor_{local_tensor_id}"
            arg_list.append(tensor_name)
            kernel_input_list.extend([
                tensor_name,
                f"{tensor_name}.shape[0]",
            ])

        # Append context argument
        arg_list.append("ctx")
        kernel_input_list.append("num_warps=8")
        kernel_input_list.append("num_stages=2")
        args_def = ",\n".join(f"{INDENT}{arg}" for arg in arg_list)
        kernel_inputs = ",\n".join(f"{INDENT * 2}{arg}" for arg in kernel_input_list)

        impl_str = f"""
{kernel_str}

def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_impl(
{args_def}
):

    {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_kernel[({ctx.num_sms * 4},)](
{kernel_inputs}
    )
"""
        return impl_str.strip()
    
    else:
        arg_list = []
        for local_tensor_id in sub_graph.input_tensor_ids:
            arg_list.append(f"tensor_{local_tensor_id}")
        for local_tensor_id in sub_graph.output_tensor_ids:
            arg_list.append(f"tensor_{local_tensor_id}")
        arg_list.append("ctx")
        args_def = ",\n".join(f"{INDENT}{arg}" for arg in arg_list)
        assert len(sub_graph.op_list) == 1, "Expected exactly one operation in non-workload-scheduled sub-graph for direct implementation."
        op_impl_func = get_impl_func(sub_graph.op_list[0])
        op_impl_str = indent_block(op_impl_func(sub_graph, 0, ctx), 1)
        impl_str = f"""
def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_impl(
{args_def}
):
{op_impl_str}
""" 
        return impl_str.strip()
