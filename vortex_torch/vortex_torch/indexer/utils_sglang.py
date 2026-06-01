import torch
from vortex_torch_C import (
sglang_plan_decode, 
sglang_plan_prefill, 
sglang_plan_decode_fa3,
sglang_plan_prefill_fa3, 
Chunkwise_NH2HN_Transpose, 
Chunkwise_HN2NH_Transpose,
Chunkwise_HN2NH_Transpose_FA3
)
from typing import Tuple
from .context import Context
from .planner_sglang import get_sglang_plan_decode_v2_module

_SCHEDULE_POLICY_ALIAS_BLOCK_SIZE = 16
_SCHEDULE_POLICY_VALUES = {
    "qwen3-1.7b-0.75": [29, 29, 45, 45, 61, 61, 77, 77, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93], 
    "qwen3-1.7b-0.80": [29, 45, 61, 77, 93, 93, 125, 125, 125, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157], 
    "qwen3-1.7b-0.83": [29, 61, 77, 93, 125, 125, 157, 157, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189], 
    "qwen3-1.7b-0.86": [29, 61, 93, 93, 125, 157, 189, 221, 221, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253],
    "qwen3-1.7b-0.88": [45, 77, 125, 157, 157, 189, 221, 253, 253, 317, 317, 317, 317, 317, 317, 317, 317, 317, 317, 317],
    "qwen3-1.7b-0.90": [45, 93, 157, 189, 221, 253, 317, 317, 381, 381, 381, 381, 381, 381, 381, 381, 381, 381, 381, 381],
    "qwen3-1.7b-0.92": [61, 125, 189, 221, 253, 317, 381, 381, 445, 509, 509, 509, 509, 509, 509, 509, 509, 509, 509, 509],
    "qwen3-1.7b-0.94": [77, 157, 221, 317, 381, 445, 509, 509, 637, 765, 765, 765, 765, 765, 765, 765, 765, 765, 765, 765], 
    "qwen3-4b-0.75": [29, 29, 29, 29, 45, 61, 77, 93, 125, 125, 125, 125, 157, 157, 157, 157, 157, 157, 189, 189], 
    "qwen3-4b-0.80": [29, 29, 45, 61, 77, 93, 125, 157, 157, 157, 189, 221, 221, 221, 221, 221, 221, 221, 221, 221], 
    "qwen3-4b-0.86": [29, 45, 61, 77, 93, 125, 157, 221, 253, 253, 317, 317, 317, 317, 381, 381, 381, 381, 509, 509], 
    "qwen3-4b-0.88": [29, 61, 93, 125, 157, 189, 253, 317, 381, 445, 445, 509, 509, 509, 637, 637, 637, 637, 637, 637], 
    "qwen3-4b-0.90": [45, 77, 125, 157, 189, 253, 317, 445, 637, 765, 765, 765, 765, 765, 765, 893, 893, 893, 893, 893], 
    "qwen3-4b-0.92": [45, 93, 157, 189, 253, 381, 445, 765, 893, 1021, 1021, 1149, 1149, 1149, 1149, 1149, 1149, 1149, 1149, 1149], 
    "qwen3-8b-0.75": [29, 29, 29, 45, 45, 77, 77, 77, 93, 93, 93, 93, 125, 125, 125, 125, 125, 125, 157, 157], 
    "qwen3-8b-0.80": [29, 29, 45, 61, 77, 93, 125, 125, 157, 157, 189, 189, 189, 221, 221, 221, 221, 253, 253, 253], 
    "qwen3-8b-0.86": [29, 45, 61, 77, 125, 157, 189, 253, 253, 253, 253, 317, 317, 317, 317, 317, 317, 385, 445, 445], 
    "qwen3-8b-0.87": [29, 61, 77, 93, 157, 189, 221, 317, 317, 317, 317, 381, 381, 381, 381, 381, 445, 445, 509, 509], 
    "qwen3-8b-0.92": [45, 93, 125, 189, 253, 445, 509, 637, 765, 765, 765, 893, 893, 893, 893, 893, 1021, 1021, 1021, 1021], 
    "qwen3-14b-0.86": [29, 45, 77, 125, 253, 381, 381, 381, 381, 381, 381, 381, 381, 381, 381, 509, 637, 637, 893, 893], 
} 

def _build_schedule_policy(values: list[int]) -> str:
    lines = ["const int offset = block_reserved_bos + block_reserved_eos;"]
    # Alias presets are tuned for token thresholds spaced every 2000 tokens.
    # The planner compares against cached_block_len, so hard-code the equivalent
    # block thresholds for block_size == 16 here.
    for upper_bound, value in zip(range(125, 2500, 125), values[:-1], strict=True):
        lines.append(f"if (cached_block_len < {upper_bound}) return {value} + offset;")
    lines.append(f"return {values[-1]} + offset;")
    return "\n".join(lines) 


SCHEDULE_POLICY_ALIASES = {
    name: _build_schedule_policy(values) for name, values in _SCHEDULE_POLICY_VALUES.items()
}


def resolve_schedule_policy(policy: str | None) -> str | None:
    if policy is None:
        return None 
    policy_key = policy.strip()
    if policy_key not in SCHEDULE_POLICY_ALIASES:
        raise ValueError(
            "Unknown vortex_schedule_policy "
            f"{policy!r}. Expected one of: {sorted(SCHEDULE_POLICY_ALIASES)}"
        )
    print("Policy selected: ", SCHEDULE_POLICY_ALIASES[policy_key], flush = True) 
    return SCHEDULE_POLICY_ALIASES[policy_key]


def assert_schedule_policy_alias_block_size(policy: str | None, block_size: int) -> None:
    if policy is None:
        return
    if policy.strip() not in SCHEDULE_POLICY_ALIASES:
        return
    assert block_size == _SCHEDULE_POLICY_ALIAS_BLOCK_SIZE, (
        "Named vortex_schedule_policy aliases assume "
        f"vortex_block_size={_SCHEDULE_POLICY_ALIAS_BLOCK_SIZE}, got {block_size}. "
        "Use a raw policy body for other block sizes."
    )


def get_decode_planner(policy: str = None):
    policy = resolve_schedule_policy(policy)

    module = get_sglang_plan_decode_v2_module(
        policy_body=policy,
        verbose=True,
        fallback_to_default=True,
    )
    def plan_decode(
        cached_seq_lens: torch.Tensor,
        req_to_token: torch.Tensor,
        req_indices: torch.Tensor,
        ctx: Context
    ):
        module.sglang_plan_decode_v2(
            cached_seq_lens,
            ctx.dense_kv_indptr,
            ctx.dense_kv_indices,
            ctx.sparse_kv_indptr,
            ctx.sparse_kv_indices,
            ctx.kv_last_page_len,
            req_to_token,
            req_indices,
            ctx.winfo_q_indices,
            ctx.winfo_kv_offsets,
            ctx.winfo_kv_lens,
            ctx.winfo_num_workloads,
            ctx.winfo_chunk_size,
            ctx.page_size,
            ctx.block_size,
            ctx.num_kv_heads,
            ctx.topk_val,
            ctx.topk_ratio,
            ctx.block_reserved_bos,
            ctx.block_reserved_eos,
            ctx.workload_chunk_size
        )
        
        ctx.set_batch_size(cached_seq_lens.shape[0])
    
    return plan_decode



def plan_prefill(
cached_seq_lens: torch.Tensor,
dense_kv_indptr: torch.Tensor,
dense_kv_indices: torch.Tensor,
input_seq_lens: torch.Tensor,
qo_indptr_ragged: torch.Tensor,
qo_indptr_paged: torch.Tensor,
kv_last_page_len: torch.Tensor,
req_to_token: torch.Tensor,
req_indices: torch.Tensor,
batch_table: torch.Tensor,
page_size: int,
num_kv_heads: int
):
    sglang_plan_prefill(
        cached_seq_lens,
        dense_kv_indptr,
        dense_kv_indices,
        input_seq_lens,
        qo_indptr_ragged,
        qo_indptr_paged,
        kv_last_page_len,
        req_to_token,
        req_indices,
        batch_table,
        page_size,
        num_kv_heads
    )


def chunkwise_nh2hn_transpose(
x: torch.Tensor,
indptr: torch.Tensor,
batch_table: torch.Tensor,
num_qo_heads: int,
num_kv_heads: int,
head_dim: int,
) -> torch.Tensor:
    
    
    x_t = Chunkwise_NH2HN_Transpose(
        x, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim
    )
    
    return x_t




def chunkwise_hn2nh_transpose(
x: torch.Tensor,
y: torch.Tensor,
indptr: torch.Tensor,
batch_table: torch.Tensor,
num_qo_heads: int,
num_kv_heads: int,
head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    x_t, y_t = Chunkwise_HN2NH_Transpose(
            x, y, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim
        )
    
    return x_t, y_t


def plan_prefill_fa3(
cached_seq_lens: torch.Tensor,
cu_seqlens_q: torch.Tensor,
req_to_token: torch.Tensor,
req_indices: torch.Tensor,
page_table: torch.Tensor,
batch_table: torch.Tensor,
page_size: int,
num_kv_heads: int
):
    sglang_plan_prefill_fa3(
        cached_seq_lens,
        cu_seqlens_q,
        req_to_token,
        req_indices,
        page_table,
        batch_table,
        page_size,
        num_kv_heads
    )
    

def plan_decode_fa3(
cached_seq_lens: torch.Tensor,
req_to_token: torch.Tensor,
req_indices: torch.Tensor,
dense_page_table: torch.Tensor,
dense_cache_seqlens: torch.Tensor,
sparse_page_table: torch.Tensor,
sparse_cache_seqlens: torch.Tensor,
ctx: Context
):
    sglang_plan_decode_fa3(
        cached_seq_lens,
        ctx.dense_kv_indptr,
        ctx.dense_kv_indices,
        ctx.sparse_kv_indptr,
        ctx.sparse_kv_indices,
        dense_page_table,
        dense_cache_seqlens,
        sparse_page_table,
        sparse_cache_seqlens,
        req_to_token,
        req_indices,
        ctx.winfo_q_indices,
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.winfo_chunk_size,
        ctx.page_size,
        ctx.num_kv_heads,
        ctx.topk_val,
        ctx.page_reserved_bos,
        ctx.page_reserved_eos,
        ctx.max_chunk_size,
        ctx.min_chunk_size
    )
    
    ctx.set_batch_size(cached_seq_lens.shape[0])
   

def chunkwise_hn2nh_transpose_fa3(
x: torch.Tensor,
indptr: torch.Tensor,
batch_table: torch.Tensor,
num_qo_heads: int,
num_kv_heads: int,
head_dim: int,
) -> torch.Tensor:
    
    x_t = Chunkwise_HN2NH_Transpose_FA3(
            x, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim
        )
    
    return x_t


def indices_to_page_table(
page_table: torch.Tensor,
ctx: Context
):
    pass
