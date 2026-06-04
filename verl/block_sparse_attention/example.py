import torch
from flash_attn import flash_attn_varlen_func
# from BSA.nsa import nsa_func 
from verl.block_sparse_attention.nsa import nsa_func 
dtype = torch.bfloat16
device = "cuda:0"
torch.cuda.set_device(device)
torch.manual_seed(20) 

def _infer_cu_seqlens_from_shapes(batch_size, seq_len, device): 
    """
    Build cu_seqlens for variable-length support. We assume *no* padding and
    only causal masking is used (your stated use case). So every sequence
    has the same length.

    query: (B, Hq, Tq, D)  key: (B, Hk, Tk, D)
    We need cu_seqlens over the **key length** per batch.
    """
    B = batch_size 
    Tk = seq_len 
    # [0, Tk, 2Tk, ... , B*Tk]
    return torch.arange(0, (B + 1) * Tk, step=Tk, dtype=torch.int32, device=device) 

q = torch.randn(size=(6, 16, 4024, 128), device=device, dtype=dtype, requires_grad=True)
k = torch.randn(size=(6, 8, 4024, 128), device=device, dtype=dtype, requires_grad=True)
v = torch.randn(size=(6, 8, 4024, 128), device=device, dtype=dtype, requires_grad=True)
cu_seqlens = _infer_cu_seqlens_from_shapes(batch_size=6, seq_len=4024, device=q.device) 
block_size = 16
block_counts = 32
window_offset = 0
# o = flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens, cu_seqlens, 4096, 4096, causal=True) 
torch.cuda.synchronize()
# o_nsa = nsa_func(
#     q, k, v, cu_seqlens, block_size, block_counts, window_offset=window_offset
# ) 

from verl.block_sparse_attention.bsa_backend import qwen3_nsa_attention 
# make a dummy module that contains the num_key_value_groups and layer_idx 
module = torch.nn.Module()
module.num_key_value_groups = 2 
module.layer_idx = 0 

o_nsa = qwen3_nsa_attention(
    module, 
    q, 
    k, 
    v, 
    attention_mask = None, 
    scaling = None, 
    dropout = 0.0, 
) 

from verl.block_sparse_attention.bsa_backend import _eager_attention_forward 
print(k.shape) 
o_eager = _eager_attention_forward(
    module, 
    q, 
    k, 
    v, 
    attention_mask = None, 
    scaling = None, 
    dropout = 0.0, 
) 

# compute the l2 distance between o_nsa and o_eager 
l2_distance = torch.norm(o_nsa - o_eager, p=2)
print(f"the l2 distance between o_nsa and o_eager is {l2_distance}")  
# o_nsa.sum().backward()
# print(k.grad.data) 
