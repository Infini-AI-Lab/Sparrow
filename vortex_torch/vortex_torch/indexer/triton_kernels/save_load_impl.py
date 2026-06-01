import torch
import triton
import triton.language as tl
from ..context import Context

@triton.jit
def save_rp_kernel(
x,  # [*, x_D0, x_D1]
o,  # [*, o_D0, o_D1]
indices,             # int32
winfo_offsets,     # int32
winfo_lens,        # int32
winfo_num_workloads, # int32*
max_chunk_size: tl.constexpr,
x_D0: tl.constexpr,
o_D0: tl.constexpr,
x_D1: tl.constexpr,
o_D1: tl.constexpr
):  
    
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    # Static even partitioning of [0, n_workloads)
    per = n_workloads // num_progs
    r   = n_workloads %  num_progs
    start = pid * per + tl.minimum(pid, r)
    end   = start + per + (pid < r)

    idx_ptr = tl.arange(0, max_chunk_size)
    
    x_dim0 = tl.arange(0, x_D0)
    x_dim1 = tl.arange(0, x_D1)
    
    
    o_dim0 = tl.arange(0, o_D0)
    o_dim1 = tl.arange(0, o_D1)

    
    for i in range(start, end):
        
        # Range of o rows for this workload
        _len = tl.load(winfo_lens + i)
        _off = tl.load(winfo_offsets + i)
        valid = idx_ptr < _len

        # Row indices for this chunk of o
        o_idx_i32 = tl.load(indices + _off + idx_ptr, mask=valid, other=0).to(tl.int32)

        offs_o = (
            (o_idx_i32[:, None, None] * (o_D0 * o_D1)) +
            (o_dim0[None, :, None]     * o_D1) +
            o_dim1[None, None, :]
        ).to(tl.int32)
        
        o_i_ptr = o + offs_o

        x_i_ptr = x + _off * x_D0 * x_D1 + \
                idx_ptr[:, None, None] * x_D0 * x_D1 + \
                x_dim0[None,:,None] * x_D1 + \
                x_dim1[None, None, :]

        x_i = tl.load(x_i_ptr, mask=valid[:,None,None], other=0.0)
        
        tl.store(o_i_ptr, x_i, mask=valid[:, None, None])
        


def save_rp(
x: torch.Tensor,
o: torch.Tensor,
ctx: Context
):  
    
    save_rp_kernel[(8 * ctx.num_sms,)](
        x, o, 
        ctx.dense_kv_indices,
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


def _save_rp(
x: torch.Tensor,
o: torch.Tensor,
indices: torch.Tensor,
winfo_offsets: torch.Tensor,
winfo_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: int,
num_sms: int
):  
    
    save_rp_kernel[(8 * num_sms,)](
        x, o, 
        indices,
        winfo_offsets,
        winfo_lens,
        winfo_num_workloads,
        max_chunk_size,
        x.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        o.shape[-1],
        num_warps=4, 
        num_stages=1
    )
    


@triton.jit
def load_pr_kernel(
x,  # [*, x_D0, x_D1]
o,  # [*, o_D0, o_D1]
indices,             # int32
winfo_offsets,     # int32
winfo_lens,        # int32
winfo_num_workloads, # int32*
max_chunk_size: tl.constexpr,
x_D0: tl.constexpr,
o_D0: tl.constexpr,
x_D1: tl.constexpr,
o_D1: tl.constexpr
):  
    
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    # Static even partitioning of [0, n_workloads)
    per = n_workloads // num_progs
    r   = n_workloads %  num_progs
    start = pid * per + tl.minimum(pid, r)
    end   = start + per + (pid < r)

    idx_ptr = tl.arange(0, max_chunk_size)
    
    x_dim0 = tl.arange(0, x_D0)
    x_dim1 = tl.arange(0, x_D1)
    
    
    o_dim0 = tl.arange(0, o_D0)
    o_dim1 = tl.arange(0, o_D1)

    
    for i in range(start, end):
        
        # Range of o rows for this workload
        _len = tl.load(winfo_lens + i)
        _off = tl.load(winfo_offsets + i)
        valid = idx_ptr < _len

        # Row indices for this chunk of o
        x_idx_i32 = tl.load(indices + _off + idx_ptr, mask=valid, other=0).to(tl.int32)

        offs_x = (
            (x_idx_i32[:, None, None] * (x_D0 * x_D1)) +
            (x_dim0[None, :, None]     * x_D1) +
            x_dim1[None, None, :]
        ).to(tl.int32)
        
        x_i_ptr = x + offs_x
        x_i = tl.load(x_i_ptr, mask=valid[:,None,None], other=0.0)
        
        
        o_i_ptr = o + _off * o_D0 * o_D1 + \
                idx_ptr[:, None, None] * o_D0 * o_D1 + \
                o_dim0[None,:,None] * o_D1 + \
                o_dim1[None, None, :]

        
        tl.store(o_i_ptr, x_i, mask=valid[:, None, None])
    

def load_pr(
x: torch.Tensor,
o: torch.Tensor,
ctx: Context
):  
    
    load_pr_kernel[(8 * ctx.num_sms,)](
        x, o, 
        ctx.dense_kv_indices,
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
    

def _load_pr(
x: torch.Tensor,
o: torch.Tensor,
indices: torch.Tensor,
winfo_offsets: torch.Tensor,
winfo_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: int,
num_sms: int
):  
    
    load_pr_kernel[(8 * num_sms,)](
        x, o, 
        indices,
        winfo_offsets,
        winfo_lens,
        winfo_num_workloads,
        max_chunk_size,
        x.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        o.shape[-1],
        num_warps=4, 
        num_stages=1
    )
    