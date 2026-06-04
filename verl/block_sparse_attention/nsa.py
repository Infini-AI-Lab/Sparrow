# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional, Union

import torch
import triton
import triton.language as tl

from fla.ops.attn.parallel import parallel_attn_bwd_preprocess
from fla.ops.nsa.compression import parallel_nsa_compression
from fla.ops.nsa.utils import _bitonic_merge
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets, prepare_lens, prepare_token_indices
from fla.ops.utils.op import exp, log
from fla.ops.utils.pooling import mean_pooling
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit
def parallel_nsa_kernel_topk(
    q,
    k,
    lse,
    scale,
    block_indices,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    WO: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)
  
    
    NEG_INF: tl.constexpr = -1e30
    INF: tl.constexpr = 1e30
    

    ################################
    # 2. topk selection
    ################################
    # [BC]
    b_i = tl.full([BC], NEG_INF, dtype=tl.float32)
    o_i = tl.full([BC], -1, dtype=tl.int32)
    m_i = tl.arange(0, BC) < BC//2

    IC = i_t // BS
    for i_c in range(0, IC + 1, BC):
        o_c = i_c + tl.arange(0, BC)

        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H*K), (0, i_c), (BK, BC), (0, 1))
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [G, BC]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where(o_c < IC, b_s, NEG_INF)
        # [G, BC]
        # the 1st and the last 2 blocks are always selected
        
        b_p = tl.where((o_c == 0) | ((o_c == IC - 1) | (o_c == IC)), INF, b_s)
        
        # the importance scores of the current block
        # [BC]
        b_i, b_ip = tl.sum(b_p, 0), b_i
        # blocks with index < 0 will be skipped
        o_i, o_ip = tl.where(o_c <= IC, o_c, -1), o_i

        n_dims: tl.constexpr = tl.standard._log2(b_i.shape[0])
        for i in tl.static_range(1, n_dims):
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, n_dims)

        if i_c != 0:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, False, n_dims)
            b_i_new = b_ip * m_i + b_i * (1 - m_i)
            o_i_new = o_ip * m_i + o_i * (1 - m_i)
            b_i, o_i = _bitonic_merge(b_i_new, o_i_new.to(tl.int32), n_dims, True, n_dims)
        else:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, True, n_dims)

    m_top = tl.arange(0, BC//S) == 0
    b_top = tl.sum(m_top[:, None] * tl.reshape(o_i, [BC//S, S]), 0)

    p_b = tl.make_block_ptr(block_indices + (bos + i_t) * H*S, (H*S,), (1,), (i_h * S,), (S,), (0,))
    tl.store(p_b, b_top.to(p_b.dtype.element_ty))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.jit
def parallel_nsa_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H*S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)

    NEG_INF: tl.constexpr = torch.finfo(torch.float32).min
    
    b_m = tl.full([G], NEG_INF, dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    b_mask = tl.zeros([1], dtype=tl.int1)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        b_mask = b_mask | (i_s <= i_t and i_s >= 0)
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BS, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            # [G, BS]
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, NEG_INF)

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = exp(b_mp - b_m)
            # [G, BS]
            b_p = exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            # [G, BV]
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

            b_mp = b_m
    b_o = b_o / b_acc[:, None]
    b_m += log(b_acc)
    b_o = tl.where(b_mask, b_o, 0.0)
    b_m = tl.where(b_mask, b_m, NEG_INF)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


@triton.heuristics({
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor)
})
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_kernel_mask(
    block_indices,
    block_counts,
    block_mask,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr
):
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S

    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(block_counts + i_b * T * H + i_t * H + i_h)
    else:
        b_m = b_i * BS <= i_t

    if b_i < NS and b_i >= 0:
        tl.store(block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i, b_m.to(block_mask.dtype.element_ty))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor)
})
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos + i_t) * HQ*K
    do += (bos + i_t) * HQ*V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * all + bos + i_t) * HQ*K
    block_indices += (bos + i_t) * H*S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BV, BS]
            b_v = tl.load(p_v, boundary_check=(0, 1))

            # [G, BS]
            b_s = tl.dot(b_q, b_k)
            b_p = exp(b_s - b_lse[:, None])
            b_p = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p, 0)

            # [G, BV] @ [BV, BS] -> [G, BS]
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            # [G, BS] @ [BS, BK] -> [G, BK]
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    block_mask,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(chunk_indices + i_s * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_v * all * H + bos * H + i_h) * K, (T, K), (H*K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))

    # [BS, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    # [BS, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i in range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * H*M + i_h * M + i_s)
        if b_m:
            p_q = tl.make_block_ptr(q + (bos + i) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
            # [G, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)

            p_do = tl.make_block_ptr(do + (bos + i) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
            p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            # [G, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [G]
            b_lse = tl.load(p_lse)
            b_delta = tl.load(p_delta)
            # [BS, G]
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= (i_s * BS + tl.arange(0, BS)))[:, None], b_p, 0)
            # [BS, G] @ [G, BV] -> [BS, BV]
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            # [BS, BV] @ [BV, G] -> [BS, G]
            b_dp = tl.dot(b_v, tl.trans(b_do))
            # [BS, G]
            b_ds = b_p * (b_dp - b_delta[None, :])
            # [BS, G] @ [G, BK] -> [BS, BK]
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def parallel_nsa_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    window_offset: Optional[int] = 0
) -> torch.LongTensor:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    # the number of selected blocks for each token
    S = block_counts if isinstance(block_counts, int) else block_counts.max().item()
    S = triton.next_power_of_2(S)
    # here we set BC = BS, but beware that they can be chosen separately if required
    BS = block_size
    BC = max(BS, 2 * S)
    BK = max(triton.next_power_of_2(K), 16)
    assert BC >= 2 * S, f"BC ({BC}) must be greater than or equal to 2 * S ({S})"

    block_indices = torch.zeros(B, T, H, S, dtype=torch.int32, device=q.device)
    token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BS) if cu_seqlens is not None else None
    grid = (T, B * H)
    # the 1st and the last 2 blocks are always selected
    parallel_nsa_kernel_topk[grid](
        q=q,
        k=k,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        WO=window_offset,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        S=S,
        BC=BC,
        BS=BS,
        BK=BK
    )
    return block_indices


def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    if check_shared_mem('hopper', q.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    grid = (T, NV, B * H)
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    parallel_nsa_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        block_counts=block_counts,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
    )

    return o, lse


def parallel_nsa_block_mask(
    block_indices: torch.LongTensor,
    block_counts: Union[torch.LongTensor, int],
    cu_seqlens: torch.LongTensor,
    block_size: int,
):
    B, T, H, S = block_indices.shape
    BS = block_size
    if cu_seqlens is not None:
        NS = triton.cdiv(prepare_lens(cu_seqlens).max().item(), BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)

    parallel_nsa_kernel_mask[(T, B, H*S)](
        block_indices=block_indices,
        block_counts=block_counts,
        block_mask=block_mask,
        T=T,
        H=H,
        S=S,
        BS=BS,
        NS=NS
    )
    return block_mask


def parallel_nsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16))
    NV = triton.cdiv(V, BV)

    delta = parallel_attn_bwd_preprocess(o, do)

    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    grid = (T, NV, B * H)
    parallel_nsa_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        block_indices=block_indices,
        block_counts=block_counts,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV
    )
    dq = dq.sum(0)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BS)
        NS = len(chunk_indices)
    else:
        chunk_indices = None
        NS = triton.cdiv(T, BS)

    # [B, T, H, M]
    block_mask = parallel_nsa_block_mask(block_indices, block_counts, cu_seqlens, block_size)
    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NS, B * H)
    parallel_nsa_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        block_mask=block_mask,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        M=block_mask.shape[-1],
        BS=BS,
        BK=BK,
        BV=BV
    )
    dk = dk.sum(0)
    return dq, dk, dv


@torch.compile
class ParallelNSAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_counts, block_size, scale, cu_seqlens):
        ctx.dtype = q.dtype

        # 2-d sequence indices denoting the cu_seqlens of tokens in each sequence
        # for example, if the passed `cu_seqlens` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None

        o, lse = parallel_nsa_fwd(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices
        )
        ctx.save_for_backward(q, k, v, o, lse, block_indices, cu_seqlens, token_indices)
        ctx.block_counts = block_counts
        ctx.block_size = block_size
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, lse, block_indices, cu_seqlens, token_indices = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            block_indices=block_indices,
            block_counts=ctx.block_counts,
            block_size=ctx.block_size,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices
        )
        return dq.to(q), dk.to(k), dv.to(v), None, None, None, None, None, None, None, None


def nsa_func(
q: torch.Tensor,
k: torch.Tensor,
v: torch.Tensor,
cu_seqlens: torch.Tensor,
block_size: int,
block_counts: int,
window_offset: int = 0
):
    scale = k.shape[-1] ** -0.5
    
    G = q.shape[2] // k.shape[2]
    B, T, HQ, D = q.shape
    if G < 16:
        assert 16 % G == 0
        q = torch.repeat_interleave(q, 16 // G, dim=2)
    else:
        assert G % 16 == 0
    
    with torch.no_grad():
        k_cmp = mean_pooling(k, block_size, cu_seqlens)
        block_indices = parallel_nsa_topk(
                q=q,
                k=k_cmp,
                lse=None,
                block_counts=block_counts,
                block_size=block_size,
                scale=scale,
                cu_seqlens=cu_seqlens,
                window_offset=window_offset
            )

    o = ParallelNSAFunction.apply(q, k, v, block_indices, block_counts, block_size, scale, cu_seqlens)
    
    if G < 16:
        o = o.reshape(B, T, HQ, 16 // G, D).mean(dim=-2)
    
    return o