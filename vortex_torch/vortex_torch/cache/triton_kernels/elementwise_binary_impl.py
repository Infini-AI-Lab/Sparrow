import torch
import triton
import triton.language as tl
from ..context import Context
from ...utils import ElementwiseBinaryOpType

@triton.jit
def elementwise_binary_ppp_kernel(
x, y, output, loc,
x_D0: tl.constexpr,
x_D1: tl.constexpr,
y_D0: tl.constexpr,
y_D1: tl.constexpr,
o_D0: tl.constexpr,
o_D1: tl.constexpr,
NUM_KV_HEAD: tl.constexpr,
PAGE_SIZE: tl.constexpr,
OP_TYPE: tl.constexpr,  # 0:Maximum, 1:Minimum, 2:Add, 3:Mul
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
    y_offset = page_id * y_D0 * y_D1
    o_offset = page_id * o_D0 * o_D1
    
    x_i = tl.load(x + x_offset + tl.arange(0, x_D0)[:, None] * x_D1 + tl.arange(0, x_D1)[None, :])
    y_i = tl.load(y + y_offset + tl.arange(0, y_D0)[:, None] * y_D1 + tl.arange(0, y_D1)[None, :])
    
    alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
    beta_bf16 = tl.full((), beta, dtype=tl.bfloat16)
    
    if OP_TYPE == 0:
        o_i = tl.maximum(x_i, y_i)
            
    elif OP_TYPE == 1:
        o_i = tl.minimum(x_i, y_i)
            
    elif OP_TYPE == 2:
        o_i = tl.add(alpha_bf16 * x_i, beta_bf16 * y_i)

    elif OP_TYPE == 3:
        o_i = x_i * y_i
            
    o_i = o_i.to(tl.bfloat16)
    
    tl.store(output + o_offset + tl.arange(0, o_D0)[:, None] * o_D1 + tl.arange(0, o_D1)[None, :], o_i)


def elementwise_binary_ppp(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    elementwise_binary_ppp_kernel[(NNZ, NUM_KV_HEAD)](
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
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )
    

def _elementwise_binary_ppp(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    elementwise_binary_ppp_kernel[(NNZ, NUM_KV_HEAD)](
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
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )


@triton.jit
def elementwise_binary_rrp_kernel(
    x, y, output, loc,
    x_D0: tl.constexpr,  # rows of x
    x_D1: tl.constexpr,  # cols of x
    y_D0: tl.constexpr,  # rows of y
    y_D1: tl.constexpr,  # cols of y
    o_D0: tl.constexpr,  # rows of output
    o_D1: tl.constexpr,  # cols of output
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    OP_TYPE: tl.constexpr,  # 0:Maximum, 1:Minimum, 2:Add (alpha*x + beta*y), 3:Mul
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    """
    Layouts:
      x:      [num_tokens, NUM_KV_HEAD, x_D0, x_D1]   (token-major)
      y:      [num_tokens, NUM_KV_HEAD, y_D0, y_D1]   (token-major)
      output: [num_pages,  NUM_KV_HEAD, o_D0, o_D1]  (page-major; flattened here as pages*o_D0*o_D1)
    Trigger only on end-of-page tokens determined by `loc`.
    """

    # program ids
    token_id = tl.program_id(0)   # 0..num_tokens-1
    head_id  = tl.program_id(1)   # 0..NUM_KV_HEAD-1

    # end-of-page check
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # ----- input offsets: token-major -----
    x_offset = (token_id * NUM_KV_HEAD + head_id) * x_D0 * x_D1
    y_offset = (token_id * NUM_KV_HEAD + head_id) * y_D0 * y_D1

    # ----- output offset: page-major -----
    page_idx = token_position // PAGE_SIZE
    page_id  = page_idx * NUM_KV_HEAD + head_id
    o_offset = page_id * o_D0 * o_D1

    # 2D indices (row-major addressing)
    x_rows = tl.arange(0, x_D0)[:, None]; x_cols = tl.arange(0, x_D1)[None, :]
    y_rows = tl.arange(0, y_D0)[:, None]; y_cols = tl.arange(0, y_D1)[None, :]
    o_rows = tl.arange(0, o_D0)[:, None]; o_cols = tl.arange(0, o_D1)[None, :]

    # load full tiles (assume already broadcasted/compatible; no mask)
    x_i = tl.load(x + x_offset + x_rows * x_D1 + x_cols)
    y_i = tl.load(y + y_offset + y_rows * y_D1 + y_cols)

    alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
    beta_bf16 = tl.full((), beta, dtype=tl.bfloat16)
    
    # elementwise binary op
    if OP_TYPE == 0:        # maximum
        o_i = tl.maximum(x_i, y_i)
    elif OP_TYPE == 1:      # minimum
        o_i = tl.minimum(x_i, y_i)
    elif OP_TYPE == 2:      # alpha*x + beta*y
        o_i = tl.add(alpha_bf16 * x_i, beta_bf16 * y_i)
    else:                   # OP_TYPE == 3: multiply
        o_i = x_i * y_i

    # cast and store (page-major)
    o_i = o_i.to(tl.bfloat16)
    tl.store(output + o_offset + o_rows * o_D1 + o_cols, o_i)


def elementwise_binary_rrp(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    elementwise_binary_rrp_kernel[(NNZ, NUM_KV_HEAD)](
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
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )


def _elementwise_binary_rrp(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    elementwise_binary_rrp_kernel[(NNZ, NUM_KV_HEAD)](
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
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )
    
@triton.jit
def elementwise_binary_rrr_kernel(
    x, y, output, loc,
    x_D0: tl.constexpr,  # rows of x
    x_D1: tl.constexpr,  # cols of x
    y_D0: tl.constexpr,  # rows of y
    y_D1: tl.constexpr,  # cols of y
    o_D0: tl.constexpr,  # rows of output
    o_D1: tl.constexpr,  # cols of output
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    OP_TYPE: tl.constexpr,   # 0:Maximum, 1:Minimum, 2:Add(alpha*x+beta*y), 3:Mul
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    """
    Layouts (all token-major):
      x:      [num_tokens, NUM_KV_HEAD, x_D0, x_D1]
      y:      [num_tokens, NUM_KV_HEAD, y_D0, y_D1]
      output: [num_tokens, NUM_KV_HEAD, o_D0, o_D1]
    Assumption: upstream guarantees shapes are equal or already broadcasted.
    Only end-of-page tokens perform the op and write.
    """

    token_id = tl.program_id(0)   # 0..num_tokens-1
    head_id  = tl.program_id(1)   # 0..NUM_KV_HEAD-1

    # trigger only at end-of-page token
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # --- base offsets (token-major) ---
    x_off = (token_id * NUM_KV_HEAD + head_id) * x_D0 * x_D1
    y_off = (token_id * NUM_KV_HEAD + head_id) * y_D0 * y_D1
    o_off = (token_id * NUM_KV_HEAD + head_id) * o_D0 * o_D1

    # --- 2D indices (row-major) ---
    x_rows = tl.arange(0, x_D0)[:, None]; x_cols = tl.arange(0, x_D1)[None, :]
    y_rows = tl.arange(0, y_D0)[:, None]; y_cols = tl.arange(0, y_D1)[None, :]
    o_rows = tl.arange(0, o_D0)[:, None]; o_cols = tl.arange(0, o_D1)[None, :]

    # --- load full tiles (assume broadcastable/equal; no mask) ---
    x_i = tl.load(x + x_off + x_rows * x_D1 + x_cols)
    y_i = tl.load(y + y_off + y_rows * y_D1 + y_cols)

    # --- ops ---
    if OP_TYPE == 0:        # maximum
        o_i = tl.maximum(x_i, y_i)
    elif OP_TYPE == 1:      # minimum
        o_i = tl.minimum(x_i, y_i)
    elif OP_TYPE == 2:      # alpha*x + beta*y  (alpha,beta -> bf16 scalars via tl.full)
        alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
        beta_bf16  = tl.full((), beta,  dtype=tl.bfloat16)
        o_i = tl.add(alpha_bf16 * x_i, beta_bf16 * y_i)
    else:                   # OP_TYPE == 3: multiply
        o_i = x_i * y_i

    # cast & store
    o_i = o_i.to(tl.bfloat16)
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)


def elementwise_binary_rrr(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    elementwise_binary_rrr_kernel[(NNZ, NUM_KV_HEAD)](
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
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )


def _elementwise_binary_rrr(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    elementwise_binary_rrr_kernel[(NNZ, NUM_KV_HEAD)](
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
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )
    
@triton.jit
def elementwise_binary_ppr_kernel(
    x, y, output, loc,
    x_D0: tl.constexpr,  # rows of a page in x
    x_D1: tl.constexpr,  # cols of a page in x
    y_D0: tl.constexpr,  # rows of a page in y
    y_D1: tl.constexpr,  # cols of a page in y
    o_D0: tl.constexpr,  # rows of output tile
    o_D1: tl.constexpr,  # cols of output tile
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    OP_TYPE: tl.constexpr,  # 0:Maximum, 1:Minimum, 2:AXPBY(alpha*x + beta*y), 3:Mul
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    """
    Layouts:
      x:      [num_pages,  NUM_KV_HEAD, x_D0, x_D1]   (page-major)
      y:      [num_pages,  NUM_KV_HEAD, y_D0, y_D1]   (page-major)
      output: [num_tokens, NUM_KV_HEAD, o_D0, o_D1]   (token-major)

    Assumptions (guaranteed by upstream):
      - Shapes are either equal or pre-broadcast to the target tile sizes.
      - Pages are full tiles; no masking is required.
      - Page order of x and y matches the page index derived from loc/PAGE_SIZE.

    Behavior:
      - Each (token_id, head_id) program checks if the token is the end of a page.
      - If yes, it reads x/y for that page (page-major), applies the elementwise op,
        and writes the result to output at (token_id, head_id, : , :) (token-major).
    """

    # -----------------------
    # Program indices
    # -----------------------
    token_id = tl.program_id(0)   # 0 .. num_tokens-1
    head_id  = tl.program_id(1)   # 0 .. NUM_KV_HEAD-1

    # -----------------------
    # End-of-page trigger
    # -----------------------
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # -----------------------
    # Page indexing (for x/y, page-major)
    # -----------------------
    page_idx = token_position // PAGE_SIZE
    page_id  = page_idx * NUM_KV_HEAD + head_id

    x_off = page_id * x_D0 * x_D1
    y_off = page_id * y_D0 * y_D1

    # -----------------------
    # Output indexing (token-major)
    # -----------------------
    out_token_id = (token_id * NUM_KV_HEAD + head_id)
    o_off = out_token_id * o_D0 * o_D1

    # -----------------------
    # 2D row-major indices
    # -----------------------
    x_rows = tl.arange(0, x_D0)[:, None]; x_cols = tl.arange(0, x_D1)[None, :]
    y_rows = tl.arange(0, y_D0)[:, None]; y_cols = tl.arange(0, y_D1)[None, :]
    o_rows = tl.arange(0, o_D0)[:, None]; o_cols = tl.arange(0, o_D1)[None, :]

    # -----------------------
    # Load full pages (no mask; upstream guarantees full tiles / broadcast)
    # -----------------------
    x_i = tl.load(x + x_off + x_rows * x_D1 + x_cols)
    y_i = tl.load(y + y_off + y_rows * y_D1 + y_cols)

    # -----------------------
    # Elementwise op
    # -----------------------
    if OP_TYPE == 0:       # Maximum
        o_i = tl.maximum(x_i, y_i)
    elif OP_TYPE == 1:     # Minimum
        o_i = tl.minimum(x_i, y_i)
    elif OP_TYPE == 2:     # AXPBY: alpha*x + beta*y
        # Use bf16 scalars as requested
        alpha_bf16 = tl.full((), alpha, dtype=tl.bfloat16)
        beta_bf16  = tl.full((), beta,  dtype=tl.bfloat16)
        o_i = tl.add(alpha_bf16 * x_i, beta_bf16 * y_i)
        # For higher accuracy (optional):
        # xi32 = x_i.to(tl.float32); yi32 = y_i.to(tl.float32)
        # o_i  = (alpha * xi32 + beta * yi32)
    else:                  # OP_TYPE == 3: Multiply
        o_i = x_i * y_i

    # Cast result to bf16 before storing (to match your pipeline)
    o_i = o_i.to(tl.bfloat16)

    # -----------------------
    # Store to output (token-major)
    # -----------------------
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)


def elementwise_binary_ppr(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    elementwise_binary_ppr_kernel[(NNZ, NUM_KV_HEAD)](
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
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )
    
    

def _elementwise_binary_ppr(
x: torch.Tensor,
y: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseBinaryOpType,
alpha: float,
beta: float
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    
    elementwise_binary_ppr_kernel[(NNZ, NUM_KV_HEAD)](
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
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta
    )