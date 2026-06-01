import torch
import triton
import triton.language as tl
from ..context import Context

@triton.jit
def fill_p_kernel(
    x, loc,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    alpha: tl.constexpr,
):
    # -----------------------------
    # Program indices
    # -----------------------------
    token_id = tl.program_id(0)
    head_id  = tl.program_id(1)

    # -----------------------------
    # Trigger only on end-of-page tokens
    # -----------------------------
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # -----------------------------
    # Page-major offset for X
    # -----------------------------
    page_idx = token_position // PAGE_SIZE
    page_id  = page_idx * NUM_KV_HEAD + head_id
    x_off    = page_id * x_D0 * x_D1

    # (Optional) alignment hints if guaranteed by upstream
    # tl.multiple_of(x_D1, 16)

    # -----------------------------
    # Build 2D row-major indices for the page
    # -----------------------------
    rows = tl.arange(0, x_D0)[:, None]     # [x_D0, 1]
    cols = tl.arange(0, x_D1)[None, :]     # [1, x_D1]
    dst  = x + x_off + rows * x_D1 + cols  # [x_D0, x_D1] pointers

    # -----------------------------
    # Fill page with alpha (bf16)
    # -----------------------------
    alpha_tile = tl.full((x_D0, x_D1), alpha, dtype=tl.bfloat16)
    tl.store(dst, alpha_tile)


def fill_p(
x: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
alpha: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    fill_p_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        alpha=alpha
    )
    

def _fill_p(
x: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
alpha: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    fill_p_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        alpha=alpha
    )