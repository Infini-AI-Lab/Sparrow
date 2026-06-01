import torch
import triton
import triton.language as tl
from ..context import Context

@triton.jit
def transpose_rr_kernel(
    x,  # [*, x_D0, x_D1]  (row-major with D1 as the fastest-changing dim)
    o,  # [*, o_D0, o_D1]  where o_D0 == x_D1 and o_D1 == x_D0
    winfo_offsets,        # int32: per-workload starting row over the leading * dimension
    winfo_lens,           # int32: per-workload row count (expected <= max_chunk_size)
    winfo_num_workloads,  # int32*: number of workloads
    #
    # compile-time constants
    max_chunk_size: tl.constexpr,
    x_D0: tl.constexpr,
    o_D0: tl.constexpr,   # must equal x_D1
    x_D1: tl.constexpr,
    o_D1: tl.constexpr,   # must equal x_D0
):
    # Program / grid partitioning (1D)
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    # Static even partition of [0, n_workloads)
    per = n_workloads // num_progs
    r = n_workloads % num_progs
    start = pid * per + tl.minimum(pid, r)
    end = start + per + (pid < r)

    # Vectorized row indices within a chunk [0, max_chunk_size)
    row_idx = tl.arange(0, max_chunk_size)

    # Indices for the last two dims
    # x is indexed as: base + d0 * x_D1 + d1
    # We want o[..., a, b] = x[..., b, a]
    # So for loading from x with output shape [rows, o_D0, o_D1] == [rows, x_D1, x_D0],
    # address offset should be: b * x_D1 + a
    o_d0 = tl.arange(0, o_D0)  # corresponds to 'a' (will map to x's d1)
    o_d1 = tl.arange(0, o_D1)  # corresponds to 'b' (will map to x's d0)

    # Iterate workloads assigned to this program
    for i in range(start, end):
        # Workload window over the leading * dimension
        _len = tl.load(winfo_lens + i)     # number of rows in this workload
        _off = tl.load(winfo_offsets + i)  # starting row index
        valid_rows = row_idx < _len        # [max_chunk_size], masks tail rows

        # Build pointers to x, but laid out as the TRANSPOSED view:
        # base over rows:
        #   base = _off*(x_D0*x_D1) + row*(x_D0*x_D1)
        # then for the last-two-dim transpose:
        #   use b=o_d1 for x's d0 with stride x_D1  (b * x_D1)
        #   use a=o_d0 for x's d1 with stride 1     (+ a)
        x_T_ptr = (
            x
            + _off * (x_D0 * x_D1)
            + row_idx[:, None, None] * (x_D0 * x_D1)
            + o_d1[None, None, :] * x_D1   # maps to x's d0
            + o_d0[None, :, None]          # maps to x's d1
        )

        # Load transposed tile directly; invalid rows get 0.0
        # Shape: [rows, o_D0, o_D1] == [rows, x_D1, x_D0]
        o_i = tl.load(x_T_ptr, mask=valid_rows[:, None, None], other=0.0)

        # Store to output at the same row window (layout already matches o)
        o_ptr = (
            o
            + _off * (o_D0 * o_D1)
            + row_idx[:, None, None] * (o_D0 * o_D1)
            + o_d0[None, :, None] * o_D1
            + o_d1[None, None, :]
        )
        tl.store(o_ptr, o_i, mask=valid_rows[:, None, None])


def transpose_rr(
x: torch.Tensor,
o: torch.Tensor,
ctx: Context
):  
    
    transpose_rr_kernel[(8 * ctx.num_sms,)](
        x, o, 
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.max_chunk_size,
        x.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        o.shape[-1],
        num_warps=4, 
        num_stages=1
    )
    


def _transpose_rr(
x: torch.Tensor,
o: torch.Tensor,
winfo_kv_offsets: torch.Tensor,
winfo_kv_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: int,
num_sms: int
):  
    
    transpose_rr_kernel[(8 * num_sms,)](
        x, o, 
        winfo_kv_offsets,
        winfo_kv_lens,
        winfo_num_workloads,
        max_chunk_size,
        x.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        o.shape[-1],
        num_warps=4, 
        num_stages=1
    )