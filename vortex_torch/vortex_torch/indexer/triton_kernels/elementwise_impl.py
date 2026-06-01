import torch
import triton
import triton.language as tl
from ..context import Context
from typing import Literal
from ...utils import ElementwiseOpType

@triton.jit
def elementwise_rr_kernel(
    x,  # [*, x_D0, x_D1]  (row-major with D1 as the fastest-changing dim)
    o,  # [*, o_D0, o_D1]
    winfo_offsets,        # int32: per-workload starting row over the leading * dimension
    winfo_lens,           # int32: per-workload row count (expected <= max_chunk_size)
    winfo_num_workloads,  # int32*: number of workloads
    #
    # compile-time constants
    max_chunk_size: tl.constexpr,
    x_D0: tl.constexpr,
    o_D0: tl.constexpr,
    x_D1: tl.constexpr,
    o_D1: tl.constexpr,
    OP_TYPE: tl.constexpr,  # "relu" | "sigmoid" | "silu" | "abs" | "add_mul"
    #
    # scalar hyper-parameters (broadcast across elements)
    alpha: float = 1.0,
    beta: float = 1.0,
):
    
    # Program id and number of programs on axis 0 (1D launch)
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Total number of workloads
    n_workloads = tl.load(winfo_num_workloads)

    # Static even partition of [0, n_workloads)
    per = n_workloads // num_progs
    r = n_workloads % num_progs
    start = pid * per + tl.minimum(pid, r)
    end = start + per + (pid < r)

    # Vectorized row indices within a chunk [0, max_chunk_size)
    row_idx = tl.arange(0, max_chunk_size)

    # Intra-row indices for linearization: row * (D0*D1) + d0 * D1 + d1
    x_d0 = tl.arange(0, x_D0)
    x_d1 = tl.arange(0, x_D1)

    o_d0 = tl.arange(0, o_D0)
    o_d1 = tl.arange(0, o_D1)

    # Convert scalars to bf16 so broadcasting matches the tensor dtype
    alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
    beta_bf16 = tl.full((), beta, dtype=tl.bfloat16)

    # Iterate workloads assigned to this program (FOR loop as requested)
    for i in range(start, end):
        # For this workload: a contiguous window over the leading * dimension
        _len = tl.load(winfo_lens + i)      # number of rows to process
        _off = tl.load(winfo_offsets + i)   # starting row index
        valid_rows = row_idx < _len         # [max_chunk_size], masks tail

        # Build pointers to the [rows, x_D0, x_D1] block in `x`
        # Layout: base + row*(x_D0*x_D1) + d0*x_D1 + d1
        x_ptr = (
            x
            + _off * (x_D0 * x_D1)
            + row_idx[:, None, None] * (x_D0 * x_D1)
            + x_d0[None, :, None] * x_D1
            + x_d1[None, None, :]
        )
        # Load with row mask; invalid rows get 0.0
        x_i = tl.load(x_ptr, mask=valid_rows[:, None, None], other=0.0)  # [rows, x_D0, x_D1], bf16

        # ----- Elementwise ops (bf16) -----
        if OP_TYPE == 0:
            # piecewise: x >= alpha ? x : beta
            o_i = tl.where(x_i >= alpha_bf16, x_i, beta_bf16)

        elif OP_TYPE == 1:
            # σ(alpha, beta; x) = 1 / (1 + exp(beta * x + alpha))
            z = (beta_bf16 * x_i + alpha_bf16)
            o_i = (1.0 / (1.0 + tl.exp(z))).to(tl.bfloat16)

        elif OP_TYPE == 2:
            # SiLU(alpha, beta; x) = x / (1 + exp(beta * x + alpha))
            z = (beta_bf16 * x_i + alpha_bf16)
            o_i = (x_i / (1.0 + tl.exp(z))).to(tl.bfloat16)

        elif OP_TYPE == 3:
            # |beta * x + alpha|
            z = beta_bf16 * x_i + alpha_bf16
            o_i = tl.abs(z)

        elif OP_TYPE == 4:
            # beta * x + alpha
            o_i = beta_bf16 * x_i + alpha_bf16

        else:
            # Fallback: write zeros for unknown OP_TYPE
            o_i = tl.zeros((max_chunk_size, o_D0, o_D1), dtype=tl.bfloat16)

        # Ensure bf16 output (harmless if already bf16)
        o_i = o_i.to(tl.bfloat16)

        # Store back to `o` at the same row window
        o_ptr = (
            o
            + _off * (o_D0 * o_D1)
            + row_idx[:, None, None] * (o_D0 * o_D1)
            + o_d0[None, :, None] * o_D1
            + o_d1[None, None, :]
        )
        tl.store(o_ptr, o_i, mask=valid_rows[:, None, None])


def elementwise_rr(
x: torch.Tensor,
o: torch.Tensor,
op_type: ElementwiseOpType,
alpha: float,
beta: float,
ctx: Context
):  
    
    elementwise_rr_kernel[(8 * ctx.num_sms,)](
        x, o, 
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.max_chunk_size,
        x.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        o.shape[-1],
        op_type.value,
        alpha,
        beta,
        num_warps=4, 
        num_stages=1
    )


def _elementwise_rr(
x: torch.Tensor,
o: torch.Tensor,
winfo_offsets: torch.Tensor,
winfo_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: int,
op_type: ElementwiseOpType,
alpha: float,
beta: float,
num_sms :int
):  
    
    elementwise_rr_kernel[(8 * num_sms,)](
        x, o, 
        winfo_offsets,
        winfo_lens,
        winfo_num_workloads,
        max_chunk_size,
        x.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        o.shape[-1],
        op_type.value,
        alpha,
        beta,
        num_warps=4, 
        num_stages=1
    )