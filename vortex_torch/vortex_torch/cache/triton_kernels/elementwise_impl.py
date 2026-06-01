import torch
import triton
import triton.language as tl
from ..context import Context
from ...utils import ElementwiseOpType

@triton.jit
def elementwise_pp_kernel(
x, output, loc,
x_D0: tl.constexpr,
x_D1: tl.constexpr,
o_D0: tl.constexpr,
o_D1: tl.constexpr,
NUM_KV_HEAD: tl.constexpr,
PAGE_SIZE: tl.constexpr,
OP_TYPE: tl.constexpr,
alpha: tl.constexpr,
beta: tl.constexpr,
):  
    
    token_id = tl.program_id(0)
    head_id  = tl.program_id(1)

    token_position = tl.load(loc + token_id)

    if (token_position + 1) % PAGE_SIZE != 0:
        return

    page_id  = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id
    x_offset = page_id * x_D0 * x_D1
    o_offset = page_id * o_D0 * o_D1
    
    x_i = tl.load(x + x_offset + tl.arange(0, x_D0)[:, None] * x_D1 + tl.arange(0, x_D1)[None, :])

    alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
    beta_bf16 = tl.full((), beta, dtype=tl.bfloat16)
    
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

            
    o_i = o_i.to(tl.bfloat16)
    
    tl.store(output + o_offset + tl.arange(0, o_D0)[:, None] * o_D1 + tl.arange(0, o_D1)[None, :], o_i)


def elementwise_pp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    elementwise_pp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )
    

def _elementwise_pp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    elementwise_pp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )
        

@triton.jit
def elementwise_rp_kernel(
    x, output, loc,
    x_D0: tl.constexpr,   # rows of x tile
    x_D1: tl.constexpr,   # cols of x tile
    o_D0: tl.constexpr,   # rows of output tile
    o_D1: tl.constexpr,   # cols of output tile
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    OP_TYPE: tl.constexpr,  # 0: piecewise(x>=alpha?x:beta), 1: sigmoid(affine), 2: silu-like, 3: abs(affine), 4: affine
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    # -----------------------------
    # Program indices
    # -----------------------------
    token_id = tl.program_id(0)  # token-major index
    head_id  = tl.program_id(1)  # head index

    # -----------------------------
    # Trigger only on end-of-page tokens
    # -----------------------------
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # -----------------------------
    # Input offset: token-major (RAGGED)
    # -----------------------------
    x_off = (token_id * NUM_KV_HEAD + head_id) * x_D0 * x_D1

    # -----------------------------
    # Output offset: page-major (PAGED)
    # -----------------------------
    page_idx = token_position // PAGE_SIZE
    page_id  = page_idx * NUM_KV_HEAD + head_id
    o_off    = page_id * o_D0 * o_D1

    # (Optional) alignment hints if guaranteed by upstream
    # tl.multiple_of(x_D1, 16)
    # tl.multiple_of(o_D1, 16)

    # -----------------------------
    # Build 2D row-major indices and load tiles
    # -----------------------------
    x_rows = tl.arange(0, x_D0)[:, None]
    x_cols = tl.arange(0, x_D1)[None, :]
    x_i = tl.load(x + x_off + x_rows * x_D1 + x_cols)  # assumes full tile; no mask

    # -----------------------------
    # Materialize alpha/beta as bf16 scalars
    # -----------------------------
    alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
    beta_bf16  = tl.full((), beta,  dtype=tl.bfloat16)

    # -----------------------------
    # Elementwise ops
    # -----------------------------
    if OP_TYPE == 0:
        # piecewise: x >= alpha ? x : beta
        o_i = tl.where(x_i >= alpha_bf16, x_i, beta_bf16)

    elif OP_TYPE == 1:
        # sigma(alpha, beta; x) = 1 / (1 + exp(beta * x + alpha))
        z = beta_bf16 * x_i + alpha_bf16
        o_i = (1.0 / (1.0 + tl.exp(z))).to(tl.bfloat16)

    elif OP_TYPE == 2:
        # silu-like(alpha, beta; x) = x / (1 + exp(beta * x + alpha))
        z = beta_bf16 * x_i + alpha_bf16
        o_i = (x_i / (1.0 + tl.exp(z))).to(tl.bfloat16)

    elif OP_TYPE == 3:
        # abs(beta * x + alpha)
        z = beta_bf16 * x_i + alpha_bf16
        o_i = tl.abs(z)

    else:  # OP_TYPE == 4
        # affine: beta * x + alpha
        o_i = beta_bf16 * x_i + alpha_bf16

    # Ensure bf16 output
    o_i = o_i.to(tl.bfloat16)

    # -----------------------------
    # Store to output page (PAGED)
    # -----------------------------
    o_rows = tl.arange(0, o_D0)[:, None]
    o_cols = tl.arange(0, o_D1)[None, :]
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)


def elementwise_rp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    elementwise_rp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )


def _elementwise_rp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    elementwise_rp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )
    
@triton.jit
def elementwise_pr_kernel(
    x, output, loc,
    x_D0: tl.constexpr,  # rows of x page
    x_D1: tl.constexpr,  # cols of x page
    o_D0: tl.constexpr,  # rows of output tile
    o_D1: tl.constexpr,  # cols of output tile
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    OP_TYPE: tl.constexpr,  # 0: piecewise(x>=alpha?x:beta), 1: sigmoid(affine), 2: silu-like, 3: abs(affine), 4: affine
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    # -----------------------------
    # Program indices
    # -----------------------------
    token_id = tl.program_id(0)  # token-major index
    head_id  = tl.program_id(1)  # head index

    # -----------------------------
    # Trigger only on end-of-page tokens
    # -----------------------------
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # -----------------------------
    # Page-major indexing for x (PAGED)
    # -----------------------------
    page_idx = token_position // PAGE_SIZE
    page_id  = page_idx * NUM_KV_HEAD + head_id
    x_off    = page_id * x_D0 * x_D1

    # -----------------------------
    # Token-major indexing for output (RAGGED)
    # -----------------------------
    out_token_lin = (token_id * NUM_KV_HEAD + head_id)
    o_off         = out_token_lin * o_D0 * o_D1

    # (Optional) alignment hints if guaranteed by upstream
    # tl.multiple_of(x_D1, 16)
    # tl.multiple_of(o_D1, 16)

    # -----------------------------
    # Build 2D row-major indices and load x page
    # -----------------------------
    x_rows = tl.arange(0, x_D0)[:, None]
    x_cols = tl.arange(0, x_D1)[None, :]
    x_i = tl.load(x + x_off + x_rows * x_D1 + x_cols)  # assumes full page; no mask

    # -----------------------------
    # Materialize alpha/beta as bf16 scalars
    # -----------------------------
    alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
    beta_bf16  = tl.full((), beta,  dtype=tl.bfloat16)

    # -----------------------------
    # Elementwise ops (unary)
    # -----------------------------
    if OP_TYPE == 0:
        # piecewise: x >= alpha ? x : beta
        o_i = tl.where(x_i >= alpha_bf16, x_i, beta_bf16)

    elif OP_TYPE == 1:
        # sigma(alpha, beta; x) = 1 / (1 + exp(beta * x + alpha))
        z = beta_bf16 * x_i + alpha_bf16
        o_i = (1.0 / (1.0 + tl.exp(z))).to(tl.bfloat16)

    elif OP_TYPE == 2:
        # silu-like(alpha, beta; x) = x / (1.0 + exp(beta * x + alpha))
        z = beta_bf16 * x_i + alpha_bf16
        o_i = (x_i / (1.0 + tl.exp(z))).to(tl.bfloat16)

    elif OP_TYPE == 3:
        # abs(beta * x + alpha)
        z = beta_bf16 * x_i + alpha_bf16
        o_i = tl.abs(z)

    else:  # OP_TYPE == 4
        # affine: beta * x + alpha
        o_i = beta_bf16 * x_i + alpha_bf16

    # Ensure bf16 output
    o_i = o_i.to(tl.bfloat16)

    # -----------------------------
    # Store into token-major output (RAGGED)
    # -----------------------------
    o_rows = tl.arange(0, o_D0)[:, None]
    o_cols = tl.arange(0, o_D1)[None, :]
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)


def elementwise_pr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    elementwise_pr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )



def _elementwise_pr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    elementwise_pr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )
    
@triton.jit
def elementwise_rr_kernel(
    x, output, loc,
    x_D0: tl.constexpr,  # rows of x tile
    x_D1: tl.constexpr,  # cols of x tile
    o_D0: tl.constexpr,  # rows of output tile
    o_D1: tl.constexpr,  # cols of output tile
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    OP_TYPE: tl.constexpr,  # 0: piecewise(x>=alpha?x:beta), 1: sigmoid(affine), 2: silu-like, 3: abs(affine), 4: affine
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
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
    # Token-major offsets (RAGGED) for x and output
    # -----------------------------
    x_off = (token_id * NUM_KV_HEAD + head_id) * x_D0 * x_D1
    o_off = (token_id * NUM_KV_HEAD + head_id) * o_D0 * o_D1

    # (Optional) alignment hints if guaranteed by upstream
    # tl.multiple_of(x_D1, 16)
    # tl.multiple_of(o_D1, 16)

    # -----------------------------
    # Build 2D row-major indices and load x tile
    # -----------------------------
    x_rows = tl.arange(0, x_D0)[:, None]
    x_cols = tl.arange(0, x_D1)[None, :]
    x_i = tl.load(x + x_off + x_rows * x_D1 + x_cols)  # assumes full tile; no mask

    # -----------------------------
    # Materialize alpha/beta as bf16 scalars
    # -----------------------------
    alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
    beta_bf16  = tl.full((), beta,  dtype=tl.bfloat16)

    # -----------------------------
    # Elementwise unary ops
    # -----------------------------
    if OP_TYPE == 0:
        # piecewise: x >= alpha ? x : beta
        o_i = tl.where(x_i >= alpha_bf16, x_i, beta_bf16)

    elif OP_TYPE == 1:
        # sigma(alpha, beta; x) = 1 / (1 + exp(beta * x + alpha))
        z = beta_bf16 * x_i + alpha_bf16
        o_i = (1.0 / (1.0 + tl.exp(z))).to(tl.bfloat16)

    elif OP_TYPE == 2:
        # silu-like(alpha, beta; x) = x / (1.0 + exp(beta * x + alpha))
        z = beta_bf16 * x_i + alpha_bf16
        o_i = (x_i / (1.0 + tl.exp(z))).to(tl.bfloat16)

    elif OP_TYPE == 3:
        # abs(beta * x + alpha)
        z = beta_bf16 * x_i + alpha_bf16
        o_i = tl.abs(z)

    else:  # OP_TYPE == 4
        # affine: beta * x + alpha
        o_i = beta_bf16 * x_i + alpha_bf16

    # Ensure bf16 for output
    o_i = o_i.to(tl.bfloat16)

    # -----------------------------
    # Store to token-major output (RAGGED)
    # -----------------------------
    o_rows = tl.arange(0, o_D0)[:, None]
    o_cols = tl.arange(0, o_D1)[None, :]
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)


def elementwise_rr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    elementwise_rr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )


def _elementwise_rr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    elementwise_rr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )