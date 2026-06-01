import torch
import triton
import triton.language as tl
from ..context import Context

@triton.jit
def gemm_ppp_kernel(
    x, y, output, loc,
    x_D0: tl.constexpr,  # rows of x page  (== o_D1)
    x_D1: tl.constexpr,  # cols of x page  (K)
    y_D0: tl.constexpr,  # rows of y page  (== o_D0)
    y_D1: tl.constexpr,  # cols of y page  (K)
    o_D0: tl.constexpr,  # rows of output page (= y_D0)
    o_D1: tl.constexpr,  # cols of output page (= x_D0)
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,  # reduction tile size along K
):
    """
    Compute one page of O = Y @ X^T in page-major layout (PPP).
    Trigger only on end-of-page tokens determined by `loc`/PAGE_SIZE.

    Shapes per page:
      Y: [y_D0, y_D1] with K = y_D1
      X: [x_D0, x_D1] with K = x_D1
      O: [o_D0, o_D1] with o_D0 = y_D0, o_D1 = x_D0
    """

    # -----------------------------
    # Program indices
    # -----------------------------
    token_id = tl.program_id(0)
    head_id  = tl.program_id(1)

    # -----------------------------
    # End-of-page trigger
    # -----------------------------
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # -----------------------------
    # Page-major indexing for X, Y, O
    # -----------------------------
    page_idx = token_position // PAGE_SIZE
    page_id  = page_idx * NUM_KV_HEAD + head_id

    x_off = page_id * x_D0 * x_D1
    y_off = page_id * y_D0 * y_D1
    o_off = page_id * o_D0 * o_D1

    # -----------------------------
    # Row indices for X/Y/O pages
    # -----------------------------
    y_rows = tl.arange(0, y_D0)[:, None]   # [y_D0, 1]
    x_rows = tl.arange(0, x_D0)[:, None]   # [x_D0, 1]
    o_rows = tl.arange(0, o_D0)[:, None]   # [o_D0, 1] == [y_D0, 1]
    o_cols = tl.arange(0, o_D1)[None, :]   # [1, o_D1] == [1, x_D0]

    # -----------------------------
    # Accumulator in fp32: [y_D0, x_D0]
    # -----------------------------
    acc = tl.zeros((y_D0, x_D0), dtype=tl.float32)

    # -----------------------------
    # K-loop with tail masking (for version)
    # -----------------------------
    K = y_D1  # assumed equal to x_D1 by upstream
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)          # [BK]
        k_mask = ks < K                          # [BK]
        ks_b1 = ks[None, :]                      # [1, BK]

        # Load Y_chunk: [y_D0, BK]
        y_ptr = y + y_off + y_rows * y_D1 + ks_b1
        y_chunk = tl.load(y_ptr, mask=k_mask[None, :], other=0.0)

        # Load X_chunk: [x_D0, BK]
        x_ptr = x + x_off + x_rows * x_D1 + ks_b1
        x_chunk = tl.load(x_ptr, mask=k_mask[None, :], other=0.0)

        # Upcast to fp32 for stable accumulation
        y32 = y_chunk.to(tl.float32)
        x32 = x_chunk.to(tl.float32)

        # Broadcast outer and reduce over BK:
        # [y_D0, 1, BK] * [1, x_D0, BK] -> [y_D0, x_D0, BK] -> sum(axis=2)
        partial = tl.sum(y32[:, None, :] * x32[None, :, :], axis=2)
        acc += partial

    # -----------------------------
    # Cast to bf16 and store O page
    # -----------------------------
    o_i = acc.to(tl.bfloat16)
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)


def gemm_ppp(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    gemm_ppp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        y=y,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        y_D0=y.shape[1],
        y_D1=y.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        BLOCK_K=32
    )


def _gemm_ppp(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    gemm_ppp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        y=y,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        y_D0=y.shape[1],
        y_D1=y.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        BLOCK_K=32
    )


@triton.jit
def gemm_ppr_kernel(
    x, y, output, loc,
    x_D0: tl.constexpr,  # rows of X page  (== o_D1)
    x_D1: tl.constexpr,  # cols of X page  (K)
    y_D0: tl.constexpr,  # rows of Y page  (== o_D0)
    y_D1: tl.constexpr,  # cols of Y page  (K)
    o_D0: tl.constexpr,  # rows of O page (= y_D0)
    o_D1: tl.constexpr,  # cols of O page (= x_D0)
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,  # reduction tile size along K
):
    """
    Compute one tile of O = Y @ X^T with PPR layout:
      X: page-major  [x_D0, x_D1] per page
      Y: page-major  [y_D0, y_D1] per page
      O: token-major [o_D0, o_D1] per token, where o_D0 = y_D0, o_D1 = x_D0

    Trigger:
      - Only end-of-page tokens perform the computation (derived from `loc` and PAGE_SIZE).

    Method:
      - Accumulate along K in chunks of BLOCK_K using broadcast-then-sum with tail masking.
      - Accumulate in fp32, cast to bf16 for store.
    """

    # -----------------------------
    # Program indices
    # -----------------------------
    token_id = tl.program_id(0)  # token-major index
    head_id  = tl.program_id(1)  # head index

    # -----------------------------
    # End-of-page trigger
    # -----------------------------
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # -----------------------------
    # Page-major indexing for X, Y (read)
    # -----------------------------
    page_idx = token_position // PAGE_SIZE
    page_id  = page_idx * NUM_KV_HEAD + head_id

    x_off = page_id * x_D0 * x_D1
    y_off = page_id * y_D0 * y_D1

    # -----------------------------
    # Token-major indexing for O (write)
    # -----------------------------
    out_lin = (token_id * NUM_KV_HEAD + head_id)
    o_off   = out_lin * o_D0 * o_D1

    # -----------------------------
    # Row indices for X/Y/O tiles
    # -----------------------------
    y_rows = tl.arange(0, y_D0)[:, None]   # [y_D0, 1]
    x_rows = tl.arange(0, x_D0)[:, None]   # [x_D0, 1]
    o_rows = tl.arange(0, o_D0)[:, None]   # [o_D0, 1] == [y_D0, 1]
    o_cols = tl.arange(0, o_D1)[None, :]   # [1, o_D1] == [1, x_D0]

    # -----------------------------
    # Accumulator in fp32: [y_D0, x_D0]
    # -----------------------------
    acc = tl.zeros((y_D0, x_D0), dtype=tl.float32)

    # -----------------------------
    # K-loop with tail masking (for syntax)
    # -----------------------------
    K = y_D1  # assumed equal to x_D1 by upstream
    for k0 in range(0, K, BLOCK_K):
        ks     = k0 + tl.arange(0, BLOCK_K)  # [BK]
        k_mask = ks < K                      # [BK]
        ks_b1  = ks[None, :]                 # [1, BK]

        # Load Y chunk: [y_D0, BK]
        y_ptr   = y + y_off + y_rows * y_D1 + ks_b1
        y_chunk = tl.load(y_ptr, mask=k_mask[None, :], other=0.0)

        # Load X chunk: [x_D0, BK]
        x_ptr   = x + x_off + x_rows * x_D1 + ks_b1
        x_chunk = tl.load(x_ptr, mask=k_mask[None, :], other=0.0)

        # Upcast to fp32 for stable accumulation
        y32 = y_chunk.to(tl.float32)         # [y_D0, BK]
        x32 = x_chunk.to(tl.float32)         # [x_D0, BK]

        # Broadcast outer and reduce over BK:
        # [y_D0, 1, BK] * [1, x_D0, BK] -> [y_D0, x_D0, BK] -> sum(axis=2)
        partial = tl.sum(y32[:, None, :] * x32[None, :, :], axis=2)
        acc += partial

    # -----------------------------
    # Cast and store to token-major output
    # -----------------------------
    o_i = acc.to(tl.bfloat16)
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)

def gemm_ppr(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    gemm_ppr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        y=y,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        y_D0=y.shape[1],
        y_D1=y.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        BLOCK_K=32
    )


def _gemm_ppr(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    gemm_ppr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        y=y,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        y_D0=y.shape[1],
        y_D1=y.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        BLOCK_K=32
    )

@triton.jit
def gemm_rrp_kernel(
    x, y, output, loc,
    x_D0: tl.constexpr,  # rows of X tile  (== o_D1)
    x_D1: tl.constexpr,  # cols of X tile  (K)
    y_D0: tl.constexpr,  # rows of Y tile  (== o_D0)
    y_D1: tl.constexpr,  # cols of Y tile  (K)
    o_D0: tl.constexpr,  # rows of O page (= y_D0)
    o_D1: tl.constexpr,  # cols of O page (= x_D0)
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,  # reduction tile size along K
):
    """
    Compute one page of O = Y @ X^T with RRP layout:
      X: token-major  [x_D0, x_D1] per (token, head)
      Y: token-major  [y_D0, y_D1] per (token, head)
      O: page-major   [o_D0, o_D1] per (page, head), where o_D0 = y_D0, o_D1 = x_D0

    Trigger:
      - Only end-of-page tokens perform the computation (from `loc` and PAGE_SIZE).

    Method:
      - Accumulate along K in chunks of BLOCK_K using broadcast-then-sum with tail masking.
      - Accumulate in fp32 for stability, cast to bf16 for store.
      - Assumes upstream guarantees y_D1 == x_D1 and shapes are valid.
    """

    # -----------------------------
    # Program indices
    # -----------------------------
    token_id = tl.program_id(0)  # token-major linear index
    head_id  = tl.program_id(1)  # head index

    # -----------------------------
    # End-of-page trigger
    # -----------------------------
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # -----------------------------
    # Token-major indexing for X/Y (RAGGED)
    # -----------------------------
    in_lin = (token_id * NUM_KV_HEAD + head_id)
    x_off  = in_lin * x_D0 * x_D1
    y_off  = in_lin * y_D0 * y_D1

    # -----------------------------
    # Page-major indexing for O (PAGED)
    # -----------------------------
    page_idx = token_position // PAGE_SIZE
    page_lin = page_idx * NUM_KV_HEAD + head_id
    o_off    = page_lin * o_D0 * o_D1

    # -----------------------------
    # Row indices for X/Y/O tiles
    # -----------------------------
    y_rows = tl.arange(0, y_D0)[:, None]   # [y_D0, 1]
    x_rows = tl.arange(0, x_D0)[:, None]   # [x_D0, 1]
    o_rows = tl.arange(0, o_D0)[:, None]   # [o_D0, 1] == [y_D0, 1]
    o_cols = tl.arange(0, o_D1)[None, :]   # [1, o_D1] == [1, x_D0]

    # -----------------------------
    # Accumulator in fp32: [y_D0, x_D0]
    # -----------------------------
    acc = tl.zeros((y_D0, x_D0), dtype=tl.float32)

    # -----------------------------
    # K-loop with tail masking (for syntax)
    # -----------------------------
    K = y_D1  # assumed equal to x_D1 by upstream
    for k0 in range(0, K, BLOCK_K):
        ks     = k0 + tl.arange(0, BLOCK_K)  # [BK]
        k_mask = ks < K                      # [BK]
        ks_b1  = ks[None, :]                 # [1, BK]

        # Load Y chunk: [y_D0, BK]
        y_ptr   = y + y_off + y_rows * y_D1 + ks_b1
        y_chunk = tl.load(y_ptr, mask=k_mask[None, :], other=0.0)

        # Load X chunk: [x_D0, BK]
        x_ptr   = x + x_off + x_rows * x_D1 + ks_b1
        x_chunk = tl.load(x_ptr, mask=k_mask[None, :], other=0.0)

        # Upcast to fp32 for stable accumulation
        y32 = y_chunk.to(tl.float32)         # [y_D0, BK]
        x32 = x_chunk.to(tl.float32)         # [x_D0, BK]

        # Broadcast outer and reduce over BK:
        # [y_D0, 1, BK] * [1, x_D0, BK] -> [y_D0, x_D0, BK] -> sum(axis=2)
        partial = tl.sum(y32[:, None, :] * x32[None, :, :], axis=2)
        acc += partial

    # -----------------------------
    # Cast and store to page-major O
    # -----------------------------
    o_i = acc.to(tl.bfloat16)
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)


def gemm_rrp(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    gemm_rrp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        y=y,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        y_D0=y.shape[1],
        y_D1=y.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        BLOCK_K=32
    )


def _gemm_rrp(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    gemm_rrp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        y=y,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        y_D0=y.shape[1],
        y_D1=y.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        BLOCK_K=32
    )

@triton.jit
def gemm_rrr_kernel(
    x, y, output, loc,
    x_D0: tl.constexpr,  # rows of X tile  (== o_D1)
    x_D1: tl.constexpr,  # cols of X tile  (K)
    y_D0: tl.constexpr,  # rows of Y tile  (== o_D0)
    y_D1: tl.constexpr,  # cols of Y tile  (K)
    o_D0: tl.constexpr,  # rows of O tile (= y_D0)
    o_D1: tl.constexpr,  # cols of O tile (= x_D0)
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,  # reduction chunk size along K
):
    """
    Compute one token-major tile of O = Y @ X^T with RRR layout:
      X: token-major  [x_D0, x_D1] per (token, head)
      Y: token-major  [y_D0, y_D1] per (token, head)
      O: token-major  [o_D0, o_D1] per (token, head), o_D0 = y_D0, o_D1 = x_D0

    Trigger:
      - Only end-of-page tokens perform the computation ((loc[token] + 1) % PAGE_SIZE == 0).

    Method:
      - Accumulate along K in steps of BLOCK_K using broadcast-then-sum with tail masking.
      - Accumulate in fp32 for numerical stability; cast to bf16 on store.
      - Upstream guarantees y_D1 == x_D1 and layout correctness.
    """

    # -----------------------------
    # Program indices
    # -----------------------------
    token_id = tl.program_id(0)  # token-major linear index
    head_id  = tl.program_id(1)  # head index

    # -----------------------------
    # End-of-page trigger
    # -----------------------------
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # -----------------------------
    # Token-major offsets (RAGGED) for X, Y, O
    # -----------------------------
    lin = token_id * NUM_KV_HEAD + head_id
    x_off = lin * x_D0 * x_D1
    y_off = lin * y_D0 * y_D1
    o_off = lin * o_D0 * o_D1

    # (Optional) alignment hints if guaranteed by upstream
    # tl.multiple_of(x_D1, 16)
    # tl.multiple_of(y_D1, 16)
    # tl.multiple_of(o_D1, 16)

    # -----------------------------
    # Row indices for X/Y/O tiles
    # -----------------------------
    y_rows = tl.arange(0, y_D0)[:, None]   # [y_D0, 1]
    x_rows = tl.arange(0, x_D0)[:, None]   # [x_D0, 1]
    o_rows = tl.arange(0, o_D0)[:, None]   # [o_D0, 1] == [y_D0, 1]
    o_cols = tl.arange(0, o_D1)[None, :]   # [1, o_D1] == [1, x_D0]

    # -----------------------------
    # Accumulator in fp32: [y_D0, x_D0]
    # -----------------------------
    acc = tl.zeros((y_D0, x_D0), dtype=tl.float32)

    # -----------------------------
    # K-loop with tail masking (for-loop form)
    # -----------------------------
    K = y_D1  # assumed equal to x_D1 by upstream
    for k0 in range(0, K, BLOCK_K):
        ks     = k0 + tl.arange(0, BLOCK_K)  # [BK]
        k_mask = ks < K                      # [BK]
        ks_b1  = ks[None, :]                 # [1, BK]

        # Load Y chunk: [y_D0, BK]
        y_ptr   = y + y_off + y_rows * y_D1 + ks_b1
        y_chunk = tl.load(y_ptr, mask=k_mask[None, :], other=0.0)

        # Load X chunk: [x_D0, BK]
        x_ptr   = x + x_off + x_rows * x_D1 + ks_b1
        x_chunk = tl.load(x_ptr, mask=k_mask[None, :], other=0.0)

        # Upcast to fp32 for stable accumulation
        y32 = y_chunk.to(tl.float32)         # [y_D0, BK]
        x32 = x_chunk.to(tl.float32)         # [x_D0, BK]

        # Broadcast outer and reduce over BK:
        # [y_D0, 1, BK] * [1, x_D0, BK] -> [y_D0, x_D0, BK] -> sum(axis=2)
        partial = tl.sum(y32[:, None, :] * x32[None, :, :], axis=2)
        acc += partial

    # -----------------------------
    # Cast and store to token-major O (RAGGED)
    # -----------------------------
    o_i = acc.to(tl.bfloat16)
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)


def gemm_rrr(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    gemm_rrr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        y=y,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        y_D0=y.shape[1],
        y_D1=y.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        BLOCK_K=32
    )


def _gemm_rrr(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    gemm_rrr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        y=y,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        y_D0=y.shape[1],
        y_D1=y.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        BLOCK_K=32
    )