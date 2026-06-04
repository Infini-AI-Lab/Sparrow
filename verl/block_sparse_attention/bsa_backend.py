# nsa_hf_backend.py
import math
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS 

import torch.nn as nn 

# ---- import your NSA path (as provided) ----
# from your_module import nsa_func  # expects (q,k,v, cu_seqlens, block_size, block_counts, window_offset=0)
# If your code is in the same file, just import directly:
from verl.block_sparse_attention.nsa import nsa_func # change to your actual module path 
from transformers.modeling_flash_attention_utils import _flash_attention_forward, flash_attn_supports_top_left_mask 


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


def qwen3_nsa_attention(
    module,
    query,            # (B, Hq, Tq, D)
    key,              # (B, Hk, Tk, D)
    value,            # (B, Hk, Tk, D)
    attention_mask,   # (B, 1, Tq, Tk) additive; we assume pure causal usage in practice
    scaling: float,   # provided by HF (1/sqrt(D)); we DO NOT use it to avoid double scaling
    dropout: float = 0.0,
    **kwargs,
):
    """
    HF attention backend wrapper for Triton NSA (block-sparse, causal only).
    Returns:
        attn_output: (B, Tq, Hq, D)
        attn_weights: None (or provide if you later implement weight dump)
    """
    # Required shapes for NSA path:
    # q -> (B, Tq, Hq, D)
    # k -> (B, Tk, Hk, D)
    # v -> (B, Tk, Hk, D)

    # Sanity
    B, Hq, Tq, D = query.shape
    _, Hk, Tk, Dk = key.shape
    assert D == Dk, "Q/K head dims must match."

    # Qwen3 passes grouped KV (Hq = Hk * G)
    G = module.num_key_value_groups
    assert Hq == Hk * G, f"Expected Hq == Hk * num_key_value_groups, got {Hq} vs {Hk}*{G}"

    # Reorder to NSA layout 
    batch_size = query.shape[0] 
    seq_len = query.shape[2] 
    
    q_nsa = query.transpose(1, 2).contiguous()   # (B, Tq, Hq, D) 
    q_nsa = q_nsa.view(-1, Hq, D).unsqueeze(0).contiguous() # (1, B*Tq, Hq, D) 
    k_nsa = key.transpose(1, 2).contiguous()     # (B, Tk, Hk, D) 
    k_nsa = k_nsa.view(-1, Hk, D).unsqueeze(0).contiguous() # (1, B*Tk, Hk, D) 
    v_nsa = value.transpose(1, 2).contiguous()   # (B, Tk, Hk, D) 
    v_nsa = v_nsa.view(-1, Hk, D).unsqueeze(0).contiguous() # (1, B*Tk, Hk, D) 

    # cu_seqlens (we assume no padding; causal-only)
    cu_seqlens = _infer_cu_seqlens_from_shapes(batch_size=batch_size, seq_len=seq_len, device=q_nsa.device) 

    # Hyperparams (tune/override via kwargs) 
    block_size = int(kwargs.get("nsa_block_size", 64)) 
    block_counts = int(kwargs.get("nsa_block_counts", 8))     # number of (past) blocks to attend per token
    window_offset = int(kwargs.get("window_offset", 0))       # optional: align with your top-k window 
    # print(f"***** block_size {block_size} block_counts {block_counts} window_offset {window_offset} kwargs {kwargs} *****") 

    # Note: your NSA computes scaling internally as D ** -0.5. We avoid double scaling.
    # Also: attention_mask is additive but we’re assuming causal-only workloads.
    # If someone passes padding masks, behavior is undefined (by design per your note).

    # If Tq != Tk (e.g., decode step), NSA still works: it will attend over Tk and produce Tq outputs.
    # nsa_func returns o: (B, Tq, Hq, D)
    o = nsa_func(
        q=q_nsa,
        k=k_nsa,
        v=v_nsa,
        cu_seqlens=cu_seqlens,
        block_size=block_size,
        block_counts=block_counts,
        window_offset=window_offset,
    ) 
    print(f"***** o.shape {o.shape} *****", flush = True) 
    o = o.squeeze(0).reshape(batch_size, Tq, Hq, D) 

    # NSA doesn’t produce weights; match HF contract
    attn_output = o  # (B, Tq, Hq, D)
    output_attentions = kwargs.get("output_attentions", False)
    attn_weights = None if not output_attentions else None  # not supported

    return attn_output, attn_weights 

def qwen3_nsa_attentiontwo( 
    module,
    query,            # (B, Hq, Tq, D)
    key,              # (B, Hk, Tk, D)
    value,            # (B, Hk, Tk, D)
    attention_mask,   # (B, 1, Tq, Tk) additive; we assume pure causal usage in practice
    scaling: float,   # provided by HF (1/sqrt(D)); we DO NOT use it to avoid double scaling
    dropout: float = 0.0,
    **kwargs,
): 
    query_states = query.transpose(1, 2).contiguous() 
    key_states = key.transpose(1, 2).contiguous() 
    value_states = value.transpose(1, 2).contiguous() 
    # cu_seqlens = torch.tensor([i * query_states.shape[1] for i in range(query_states.shape[0] + 1)], device=query_states.device, dtype=torch.int32) 
    idsposition = kwargs["position_ids"] 
    print(f"***** idsposition {idsposition.shape} *****", flush = True) 
    idsposition = torch.tensor(idsposition, device=query_states.device, dtype=torch.int32) 
    cu_seqlens = torch.where(idsposition[0] == 0)[0] 
    cu_seqlens = torch.cat([cu_seqlens, torch.tensor([query_states.shape[1]], device=query_states.device, dtype=torch.int32)]) 
    # batch_size = query_states.shape[1] // idsposition.shape[1] 
    # cu_seqlens = torch.tensor([i * idsposition.shape[1] for i in range(batch_size + 1)], device=query_states.device, dtype=torch.int32) 
    
    block_size = int(kwargs.get("nsa_block_size", 16)) 
    block_counts = int(kwargs.get("nsa_block_counts", 32))     # number of (past) blocks to attend per token
    window_offset = int(kwargs.get("window_offset", 0))       # optional: align with your top-k window 
    print(f"***** block_size {block_size} block_counts {block_counts} window_offset {window_offset} *****", flush = True) 
    
    attn_output = nsa_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens=cu_seqlens,
        block_size=block_size, 
        block_counts=block_counts, 
    ) 
    
    return attn_output, None 

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim) 

def _eager_attention_forward(
    module, 
    query, 
    key, 
    value, 
    attention_mask, 
    scaling, 
    dropout = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights 

_use_top_left_mask = flash_attn_supports_top_left_mask() 

def get_target_dtype(query, module):
    """If the query is in float32, return a target dtype compatible with flash attention. Return None otherwise."""
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            return module.config._pre_quantization_dtype
        else:
            return next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype
    return None 


def flash_attention_forward(
    module, 
    query,
    key,
    value,
    attention_mask,
    dropout, 
    scaling, 
    sliding_window, 
    softcap, 
    is_causal, 
    **kwargs, 
): 

    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = get_target_dtype(query, module)

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else module.is_causal

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        attn_implementation="flash_attention_2", 
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        **kwargs,
    ) 

    return attn_output, None

def qwen3_bsa_attention_complete(
    module,
    query,            # (B, Hq, Tq, D)
    key,              # (B, Hk, Tk, D)
    value,            # (B, Hk, Tk, D)
    attention_mask,   # (B, 1, Tq, Tk) additive; we assume pure causal usage in practice
    scaling: float,   # provided by HF (1/sqrt(D)); we DO NOT use it to avoid double scaling
    dropout: float = 0.0,
    **kwargs,
): 
    # print(f"***** attention mask {attention_mask.shape} *****", flush = True) 
    # check if attention mask is all ones 
    # print(f"***** attention mask {attention_mask} *****", flush = True) 
    # print(f"***** scaling {scaling} *****", flush = True) 
    # print(f"***** dropout {dropout} *****", flush = True) 
    # print(f"***** kwargs {kwargs} *****", flush = True) 
    # print(f"***** query {query.shape} *****", flush = True) 
    # print(f"***** key {key.shape} *****", flush = True) 
    # print(f"***** value {value.shape} *****", flush = True) 
    
    use_full_dense = getattr(module, "use_full_dense", None) 
    if use_full_dense: 
        return flash_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, softcap = None, is_causal = True, **kwargs) 
    else: 
        if module.layer_idx in [0, 1]: 
            # use eager attention 
            # print("***** use eager attention for layer 0 and 1 ******") 
            # return _eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout, **kwargs) 
            return flash_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, softcap = None, is_causal = True, **kwargs) 
        else: 
            # use NSA attention 
            # print("***** use NSA attention for layer 2 and 3 ******") 
            return qwen3_nsa_attentiontwo(module, query, key, value, attention_mask, scaling, dropout, **kwargs) 


# ---- register with HF ----
ALL_ATTENTION_FUNCTIONS["nsa_triton"] = qwen3_bsa_attention_complete 
