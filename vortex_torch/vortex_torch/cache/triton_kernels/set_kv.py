import torch
import triton
import triton.language as tl

@triton.jit
def set_kv_buffer_kernel(
    k_cache,
    v_cache,
    new_k,
    new_v,
    loc,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr
):
    
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)    
    dim = tl.arange(0, HEAD_DIM)
    
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr)
    src_v = tl.load(new_v + src_ptr)
    
    token_position = tl.load(loc + token_id)
    position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_position %  PAGE_SIZE
    
    dst_k_ptr = k_cache + position_trans * HEAD_DIM + dim
    dst_v_ptr = v_cache + position_trans * HEAD_DIM + dim
    
    tl.store(dst_k_ptr, src_k)
    tl.store(dst_v_ptr, src_v)


def set_kv_buffer_launcher(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]
    
    set_kv_buffer_kernel[(NNZ, NUM_KV_HEAD)](
        k_cache,
        v_cache,
        new_k,
        new_v,
        loc,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size
    )



@triton.jit
def set_kv_buffer_fp8_e5m2_kernel(
    k_cache,
    v_cache,
    new_k,
    new_v,
    loc,
    k_scale: tl.constexpr,
    v_scale: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)    
    dim = tl.arange(0, HEAD_DIM)
    
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr).to(tl.float32)
    src_v = tl.load(new_v + src_ptr).to(tl.float32)

    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale
    scaled_k = src_k * inv_k_scale
    scaled_v = src_v * inv_v_scale

    clamped_k = tl.minimum(tl.maximum(scaled_k, -57344.0), 57344.0)
    clamped_v = tl.minimum(tl.maximum(scaled_v, -57344.0), 57344.0)
    q_k = clamped_k.to(tl.float8e5).to(tl.uint8, bitcast=True)
    q_v = clamped_v.to(tl.float8e5).to(tl.uint8, bitcast=True)
    
    
    token_position = tl.load(loc + token_id)
    position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_position %  PAGE_SIZE
    
    dst_k_ptr = k_cache + position_trans * HEAD_DIM + dim
    dst_v_ptr = v_cache + position_trans * HEAD_DIM + dim
    
    tl.store(dst_k_ptr, q_k)
    tl.store(dst_v_ptr, q_v)



def set_kv_buffer_fp8_e5m2_launcher(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int,
    k_scale: float = None,
    v_scale: float = None,
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]

    k_scale = 1.0 if k_scale is None else k_scale
    v_scale = 1.0 if v_scale is None else v_scale

    set_kv_buffer_fp8_e5m2_kernel[(NNZ, NUM_KV_HEAD)](
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        new_k,
        new_v,
        loc,
        k_scale,
        v_scale,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size
    )


@triton.jit
def set_kv_buffer_fp8_e4m3_kernel(
    k_cache,
    v_cache,
    new_k,
    new_v,
    loc,
    k_scale: tl.constexpr,
    v_scale: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)    
    dim = tl.arange(0, HEAD_DIM)
    
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr).to(tl.float32)
    src_v = tl.load(new_v + src_ptr).to(tl.float32)

    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale
    scaled_k = src_k * inv_k_scale
    scaled_v = src_v * inv_v_scale

    clamped_k = tl.minimum(tl.maximum(scaled_k, -448.0), 448.0)
    clamped_v = tl.minimum(tl.maximum(scaled_v, -448.0), 448.0)
    q_k = clamped_k.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    q_v = clamped_v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    
    
    token_position = tl.load(loc + token_id)
    position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_position %  PAGE_SIZE
    
    dst_k_ptr = k_cache + position_trans * HEAD_DIM + dim
    dst_v_ptr = v_cache + position_trans * HEAD_DIM + dim
    
    tl.store(dst_k_ptr, q_k)
    tl.store(dst_v_ptr, q_v)



def set_kv_buffer_fp8_e4m3_launcher(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int,
    k_scale: float = None,
    v_scale: float = None,
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]

    k_scale = 1.0 if k_scale is None else k_scale
    v_scale = 1.0 if v_scale is None else v_scale

    set_kv_buffer_fp8_e4m3_kernel[(NNZ, NUM_KV_HEAD)](
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        new_k,
        new_v,
        loc,
        k_scale,
        v_scale,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size
    )

