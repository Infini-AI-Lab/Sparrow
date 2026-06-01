import torch
import triton
import triton.language as tl
from ..context import Context
from ...utils import ElementwiseBinaryOpType

@triton.jit
def elementwise_binary_rrr_kernel(
x,  # [*, x_D0, x_D1]
y,  # [*, y_D0, y_D1]
o,  # [*, o_D0, o_D1]
winfo_offsets,     # int32
winfo_lens,        # int32
winfo_num_workloads, # int32*
max_chunk_size: tl.constexpr,
x_D0: tl.constexpr,
y_D0: tl.constexpr,
o_D0: tl.constexpr,
x_D1: tl.constexpr,
y_D1: tl.constexpr,
o_D1: tl.constexpr,
OP_TYPE: tl.constexpr,
alpha: float = 1.0,
beta: float = 1.0
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
    
    y_dim0 = tl.arange(0, y_D0)
    y_dim1 = tl.arange(0, y_D1)
    
    o_dim0 = tl.arange(0, o_D0)
    o_dim1 = tl.arange(0, o_D1)
    
    alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
    beta_bf16 = tl.full((), beta, dtype=tl.bfloat16)
    
    for i in range(start, end):
        
        # Range of y rows for this workload
        _len = tl.load(winfo_lens + i)
        _off = tl.load(winfo_offsets + i)
        valid = idx_ptr < _len
        x_i_ptr = x + _off * x_D0 * x_D1 + \
                idx_ptr[:, None, None] * x_D0 * x_D1 + \
                x_dim0[None,:,None] * x_D1 + \
                x_dim1[None, None, :]

        x_i = tl.load(x_i_ptr, mask=valid[:,None,None], other=0.0)
        
        
        y_i_ptr = y + _off * y_D0 * y_D1 + \
                idx_ptr[:, None, None] * y_D0 * y_D1 + \
                y_dim0[None,:,None] * y_D1 + \
                y_dim1[None, None, :]

        y_i = tl.load(y_i_ptr, mask=valid[:,None,None], other=0.0)
        
        
        if OP_TYPE == 0:
            o_i = tl.maximum(x_i, y_i)
            
        elif OP_TYPE == 1:
            o_i = tl.minimum(x_i, y_i)
            
        elif OP_TYPE == 2:
            o_i = tl.add(alpha_bf16 * x_i, beta_bf16 * y_i)

        elif OP_TYPE == 3:
            o_i = x_i * y_i
                
        else:
            o_i = tl.zeros((max_chunk_size, o_D0, o_D1), dtype=tl.bfloat16)
            
        o_i = o_i.to(tl.bfloat16)
        
        o_i_ptr = o + _off * o_D0 * o_D1 + \
                idx_ptr[:, None, None] * o_D0 * o_D1 + \
                o_dim0[None,:,None] * o_D1 + \
                o_dim1[None, None, :]
        
        tl.store(o_i_ptr, o_i, mask=valid[:, None, None])
        
        




def elementwise_binary_rrr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float,
ctx: Context
):  
    
    elementwise_binary_rrr_kernel[(8 * ctx.num_sms,)](
        x, y, o, 
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.max_chunk_size,
        x.shape[-2], 
        y.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        y.shape[-1],
        o.shape[-1],
        op_type.value,
        alpha,
        beta,
        num_warps=4, 
        num_stages=1
    )


def _elementwise_binary_rrr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
winfo_offsets: torch.Tensor,
winfo_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: int,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float,
num_sms: int
):  
    
    elementwise_binary_rrr_kernel[(8 * num_sms,)](
        x, y, o, 
        winfo_offsets,
        winfo_lens,
        winfo_num_workloads,
        max_chunk_size,
        x.shape[-2], 
        y.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        y.shape[-1],
        o.shape[-1],
        op_type.value,
        alpha,
        beta,
        num_warps=4, 
        num_stages=1
    )


@triton.jit
def elementwise_binary_bpr_kernel(
x,  # bf16, [B, x_D0, x_D1] ; x_D1 is the fastest-changing dim
y,  # bf16, [S, y_D0, y_D1] ; y_D1 is the fastest-changing dim
o,  # bf16, [*, o_D0, o_D1] ; flat buffer (each row writes a [o_D0, o_D1] tile)
indices,             # int32
winfo_x_indices,     # int32
winfo_y_offsets,     # int32
winfo_y_lens,        # int32
winfo_num_workloads, # int32*
max_chunk_size: tl.constexpr,
x_D0: tl.constexpr,
y_D0: tl.constexpr,
o_D0: tl.constexpr, # max(x_D0, y_D0), upstream will guarantee x_D0, y_D0 are broadcastable
x_D1: tl.constexpr,
y_D1: tl.constexpr,
o_D1: tl.constexpr, # max(x_D1, y_D1), upstream will guarantee x_D1, y_D1 are broadcastable
OP_TYPE: tl.constexpr,
alpha: float = 1.0,
beta: float = 1.0
):  

    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    # Static even partitioning of [0, n_workloads)
    per = n_workloads // num_progs
    r   = n_workloads %  num_progs
    start = pid * per + tl.minimum(pid, r)
    end   = start + per + (pid < r)

    # Index vectors
    
    x_d0_ptr   = tl.arange(0, x_D0)
    y_d0_ptr   = tl.arange(0, y_D0)
    o_d0_ptr   = tl.arange(0, o_D0)
    
    x_d1_ptr   = tl.arange(0, x_D1)
    y_d1_ptr   = tl.arange(0, y_D1)
    o_d1_ptr   = tl.arange(0, o_D1)
    
    idx_ptr = tl.arange(0, max_chunk_size)

    # Stride across B for x
    x_stride = x_D0 * x_D1

    # Persistent cache: the current x[x_idx] as a whole [x_D0, D1] tile
    current_x_idx = tl.full((), -1, dtype=tl.int32)
    x_i = tl.zeros((x_D0, x_D1), dtype=tl.bfloat16)
    alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
    beta_bf16 = tl.full((), beta, dtype=tl.bfloat16)
    
    for i in range(start, end):
        # Select x for this workload
        x_idx_i32 = tl.load(winfo_x_indices + i).to(tl.int32)
        if x_idx_i32 != current_x_idx:
            x_base = (x_idx_i32 * x_stride).to(tl.int32)
            # Load x_i: [x_D0, D1] (bf16)
            x_offs = x_base + (x_d0_ptr[:, None] * x_D1 + x_d1_ptr[None, :]).to(tl.int32)
            x_i = tl.load(x + x_offs)  # bf16
            current_x_idx = x_idx_i32

        # Range of y rows for this workload
        y_len = tl.load(winfo_y_lens + i)
        y_off = tl.load(winfo_y_offsets + i)
        valid = idx_ptr < y_len

        # Row indices for this chunk of y
        y_idx_i32 = tl.load(indices + y_off + idx_ptr, mask=valid, other=0).to(tl.int32)

        # Load y_tile: [rows, y_D0, D1] (bf16)
        # Linear offset: row*y_D0*D1 + y_d0*D1 + d1
        offs_y = (
            (y_idx_i32[:, None, None] * (y_D0 * y_D1)) +
            (y_d0_ptr[None, :, None]     * y_D1) +
            y_d1_ptr[None, None, :]
        ).to(tl.int32)
        y_i = tl.load(y + offs_y, mask=valid[:, None, None], other=0.0)  # [rows, y_D0, D1], bf16

        
        # Elementwise in bf16:
        # [rows, y_D0, y_D1],  [1, x_D0, x_D1] -> [rows, o_D0, o_D1] (bf16)
        
        if OP_TYPE == 0:
            o_i = tl.maximum(x_i[None,:,:], y_i)
            
        elif OP_TYPE == 1:
            o_i = tl.minimum(x_i[None,:,:], y_i)
            
        elif OP_TYPE == 2:
            o_i = tl.add(alpha_bf16 * x_i[None,:,:], beta_bf16 * y_i)

        elif OP_TYPE == 3:
            o_i = x_i[None,:,:] * y_i
                
        else:
            o_i = tl.zeros((max_chunk_size, o_D0, o_D1), dtype=tl.bfloat16)
            
        
        # Linear output offset: row*o_D0*D1 + o_d0*D1 + d1, where row starts at y_off
        offs_o = (
            ((y_off + idx_ptr[:, None, None]) * (o_D0 * o_D1)) +
            (o_d0_ptr[None, :, None] * o_D1) +
            o_d1_ptr[None, None, :]
        ).to(tl.int32)

        tl.store(o + offs_o, o_i, mask=valid[:, None, None])


def elementwise_binary_bpr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float,
ctx: Context
):  
    
    elementwise_binary_bpr_kernel[(8 * ctx.num_sms,)](
        x, y, o, 
        ctx.dense_kv_indices,
        ctx.winfo_q_indices,
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.max_chunk_size,
        x.shape[-2], 
        y.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        y.shape[-1],
        o.shape[-1],
        op_type.value,
        alpha,
        beta,
        num_warps=4, 
        num_stages=1
    )
    

def _elementwise_binary_bpr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
indices: torch.Tensor,
winfo_x_indices: torch.Tensor,
winfo_y_offsets: torch.Tensor,
winfo_y_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: int,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float,
num_sms: int,
):  
    
    elementwise_binary_bpr_kernel[(8 * num_sms,)](
        x, y, o, 
        indices,
        winfo_x_indices,
        winfo_y_offsets,
        winfo_y_lens,
        winfo_num_workloads,
        max_chunk_size,
        x.shape[-2], 
        y.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        y.shape[-1],
        o.shape[-1],
        op_type.value,
        alpha,
        beta,
        num_warps=4, 
        num_stages=1
    )
    

@triton.jit
def elementwise_binary_rpr_kernel(
x,  # [*, x_D0, x_D1]
y,  # [*, y_D0, y_D1]
o,  # [*, o_D0, o_D1]
indices,             # int32
winfo_offsets,     # int32
winfo_lens,        # int32
winfo_num_workloads, # int32*
max_chunk_size: tl.constexpr,
x_D0: tl.constexpr,
y_D0: tl.constexpr,
o_D0: tl.constexpr,
x_D1: tl.constexpr,
y_D1: tl.constexpr,
o_D1: tl.constexpr,
OP_TYPE: tl.constexpr,
alpha: float = 1.0,
beta: float = 1.0
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
    
    y_dim0 = tl.arange(0, y_D0)
    y_dim1 = tl.arange(0, y_D1)
    
    o_dim0 = tl.arange(0, o_D0)
    o_dim1 = tl.arange(0, o_D1)

    alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
    beta_bf16 = tl.full((), beta, dtype=tl.bfloat16)
    
    for i in range(start, end):
        

        # Range of y rows for this workload
        _len = tl.load(winfo_lens + i)
        _off = tl.load(winfo_offsets + i)
        valid = idx_ptr < _len

        # Row indices for this chunk of y
        y_idx_i32 = tl.load(indices + _off + idx_ptr, mask=valid, other=0).to(tl.int32)

        offs_y = (
            (y_idx_i32[:, None, None] * (y_D0 * y_D1)) +
            (y_dim0[None, :, None]     * y_D1) +
            y_dim1[None, None, :]
        ).to(tl.int32)
        
        y_i = tl.load(y + offs_y, mask=valid[:, None, None], other=0.0)  # [rows, C, D], bf16

        x_i_ptr = x + _off * x_D0 * x_D1 + \
                idx_ptr[:, None, None] * x_D0 * x_D1 + \
                x_dim0[None,:,None] * x_D1 + \
                x_dim1[None, None, :]

        x_i = tl.load(x_i_ptr, mask=valid[:,None,None], other=0.0)
        
        if OP_TYPE == 0:
            o_i = tl.maximum(x_i, y_i)
            
        elif OP_TYPE == 1:
            o_i = tl.minimum(x_i, y_i)
            
        elif OP_TYPE == 2:
            o_i = tl.add(alpha_bf16 * x_i, beta_bf16 * y_i)

        elif OP_TYPE == 3:
            o_i = x_i * y_i
                
        else:
            o_i = tl.zeros((max_chunk_size, o_D0, o_D1), dtype=tl.bfloat16)
            
        
        o_i_ptr = o + _off * o_D0 * o_D1 + \
                idx_ptr[:, None, None] * o_D0 * o_D1 + \
                o_dim0[None,:,None] * o_D1 + \
                o_dim1[None, None, :]
        
        
        tl.store(o_i_ptr, o_i, mask=valid[:, None, None])
        


def elementwise_binary_rpr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float,
ctx: Context
):  
    
    elementwise_binary_rpr_kernel[(8 * ctx.num_sms,)](
        x, y, o, 
        ctx.dense_kv_indices,
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.max_chunk_size,
        x.shape[-2], 
        y.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        y.shape[-1],
        o.shape[-1],
        op_type.value,
        alpha,
        beta,
        num_warps=4, 
        num_stages=1
    )
    

def _elementwise_binary_rpr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
indices: torch.Tensor,
winfo_offsets: torch.Tensor,
winfo_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: int,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float,
num_sms: int
):  
    
    elementwise_binary_rpr_kernel[(8 * num_sms,)](
        x, y, o, 
        indices,
        winfo_offsets,
        winfo_lens,
        winfo_num_workloads,
        max_chunk_size,
        x.shape[-2], 
        y.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        y.shape[-1],
        o.shape[-1],
        op_type.value,
        alpha,
        beta,
        num_warps=4, 
        num_stages=1
    )
    