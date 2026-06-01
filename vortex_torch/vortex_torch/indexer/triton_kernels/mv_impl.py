import torch
import triton
import triton.language as tl
from ..context import Context

@triton.jit
def mv_bpr_kernel(
x,
y,
o,
indices,
winfo_x_indices,
winfo_y_offsets,
winfo_y_lens,
winfo_num_workloads,
workload_chunk_size: tl.constexpr,
D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    per = n_workloads // num_progs
    r = n_workloads % num_progs
    start = pid * per + tl.minimum(pid, r)
    end = start + per + (pid < r)

    
    d_ptr = tl.arange(0, D)
    idx_ptr = tl.arange(0, workload_chunk_size)

    x_stride = D

    current_x_idx = tl.full((), -1, dtype=tl.int32)
    x_i = tl.zeros((D,), dtype=tl.bfloat16)

    for i in range(start, end):
        
        x_idx_i32 = tl.load(winfo_x_indices + i).to(tl.int32)
        if x_idx_i32 != current_x_idx:
            x_base_offs = x_idx_i32 * x_stride
            x_i = tl.load(
                x + x_base_offs + d_ptr
            )
            current_x_idx = x_idx_i32

        y_len = tl.load(winfo_y_lens + i)
        y_off = tl.load(winfo_y_offsets + i)
        valid = idx_ptr < y_len

        
        y_idx_i32 = tl.load(indices + y_off + idx_ptr, mask=valid, other=0).to(tl.int32)

        row_offs = y_idx_i32 * D     # [workload_chunk_size]
        offs = row_offs[:, None] + d_ptr[None, :]             

        y_tile = tl.load(y + offs, mask=valid[:, None], other=0.0)
        
        o_i = tl.sum((y_tile * x_i[None, :]).to(tl.float32), 1)
        o_i = o_i.to(tl.bfloat16)
        tl.store(o + y_off + idx_ptr, o_i, mask=valid)



@triton.jit
def group_mv_bpr_kernel(
    x,                      # [XN, D] matrix of x vectors
    y,                      # [YN, D] matrix of y vectors
    o,                      # output buffer
    indices,                # index buffer (only first index per group is used)
    winfo_x_indices,        # [n_workloads]: which x each workload uses
    winfo_y_offsets,        # [n_workloads]: output / indices offset
    winfo_y_lens,           # [n_workloads]: number of valid y in this group
    winfo_num_workloads,    # scalar: total workloads
    XN: tl.constexpr,                     # number of rows in x
    YN: tl.constexpr,                     # number of rows in y
    W: tl.constexpr,        # max group size (tile height)
    D: tl.constexpr,        # embedding dimension (tile width)
):
    # -------------------------------------------------------
    # 1. Program-level workload partitioning
    # -------------------------------------------------------
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    # Evenly split workloads across programs
    per = n_workloads // num_progs
    r = n_workloads % num_progs
    start = pid * per + tl.minimum(pid, r)
    end = start + per + (pid < r)

    # Index inside a group (0 ... W-1)
    idx_ptr = tl.arange(0, W)

    # Row-major layout: [N, D]
    x_stride_0 = D
    x_stride_1 = 1

    y_stride_0 = D
    y_stride_1 = 1

    # Cache the last loaded x to avoid redundant memory loads
    current_x_idx = tl.full((), -1, dtype=tl.int32)
    x_i = tl.zeros((1, D,), dtype=tl.bfloat16)

    # -------------------------------------------------------
    # 2. Loop over workloads assigned to this program
    # -------------------------------------------------------
    for i in range(start, end):

        # -----------------------------------
        # Load x index for this workload
        # -----------------------------------
        x_idx_i32 = tl.load(winfo_x_indices + i).to(tl.int32)

        # Reload x only if index changed (cache reuse)
        if x_idx_i32 != current_x_idx:
            # Create a block pointer to row x[x_idx_i32, :]
            x_block_ptr = tl.make_block_ptr(
                base=x,
                shape=(XN, D),
                strides=(x_stride_0, x_stride_1),
                offsets=(x_idx_i32, 0),   # start at row x_idx_i32
                block_shape=(1, D),       # load 1 row
                order=(1, 0),
            )

            # Load row → shape [1, D]
            x_i = tl.load(
                x_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            current_x_idx = x_idx_i32

        # -----------------------------------
        # Load workload metadata
        # -----------------------------------
        y_len = tl.load(winfo_y_lens + i)
        y_off = tl.load(winfo_y_offsets + i)

        # Mask for valid elements within this group
        valid = idx_ptr < y_len

        # -----------------------------------
        # Load base y index for this group
        # -----------------------------------
        # IMPORTANT:
        # We assume y indices are contiguous:
        # indices[y_off + j] = indices[y_off] + j
        y_idx_i32 = tl.load(indices + y_off).to(tl.int32)

        # -----------------------------------
        # Load a block of y vectors: shape [W, D]
        # -----------------------------------
        y_block_ptr = tl.make_block_ptr(
            base=y,
            shape=(YN, D),
            strides=(y_stride_0, y_stride_1),
            offsets=(y_idx_i32, 0),   # starting row
            block_shape=(W, D),       # load W consecutive rows
            order=(1, 0),
        )

        y_tile = tl.load(
            y_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )  # shape: [W, D]

        # Mask out rows beyond y_len
        y_tile = tl.where(valid[:, None], y_tile, 0.0)

        # -----------------------------------
        # Compute dot products
        # -----------------------------------
        # Each row of y_tile is dotted with x_i
        o_i = tl.sum((y_tile * x_i).to(tl.float32), axis=1)
        o_i = o_i.to(tl.bfloat16)

        # -----------------------------------
        # Store results
        # -----------------------------------
        tl.store(o + y_off + idx_ptr, o_i, mask=valid)


def mv_bpr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
ctx: Context
):  
    
    group_mv_bpr_kernel[(4 * ctx.num_sms,)](
        x, y, o, 
        ctx.dense_kv_indices,
        ctx.winfo_q_indices,
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        x.shape[0], y.shape[0],
        ctx.workload_chunk_size,
        x.shape[-1], num_warps=8, num_stages=2
    )
