import torch
import triton
import triton.testing as testing
from flash_attn import flash_attn_varlen_func
from BSA.nsa import nsa_func

# 构造数据
dtype = torch.bfloat16
device = "cuda:5"
torch.cuda.set_device(device)
B, T, Hq, Hkv, D = 1, 16384, 32, 4, 128
block_size, block_counts = 32, 32

q = torch.randn((B, T, Hq, D), device=device, dtype=dtype, requires_grad=False)
k = torch.randn((B, T, Hkv, D), device=device, dtype=dtype, requires_grad=False)
v = torch.randn((B, T, Hkv, D), device=device, dtype=dtype, requires_grad=False)
cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

# 定义封装函数
def run_flash():
    return flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens, cu_seqlens, T, T, causal=True)

def run_nsa():
    return nsa_func(q, k, v, cu_seqlens, block_size, block_counts)

# Benchmark
ms_flash = testing.do_bench(run_flash, warmup=100, rep=500)
ms_nsa   = testing.do_bench(run_nsa, warmup=100, rep=500)

print(f"FlashAttention latency: {ms_flash:.3f} ms")
print(f"NSA latency:           {ms_nsa:.3f} ms")
