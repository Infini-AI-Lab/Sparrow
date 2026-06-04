from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Dict, Tuple
from functools import partial
import torch
import vortex_torch
from vortex_torch.abs import as_vtensor, FORMAT
if os.environ["SGLANG_ENABLE_TORCH_COMPILE"] == "1":
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.utils import is_sm100_supported
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.mem_cache.vtx_graph_memory_pool import VTXGraphCachePool
if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.decode import _get_range_buf, get_seq_lens

@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]

@dataclass
class PrefillMetadata:
    extend_no_prefix: bool

from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    
    unexpand_cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    
    unexpand_cu_seqlens_k: torch.Tensor = None
    
    eff_bs: int = 0


    
# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class VTXFA3AttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        self.decode_use_tensor_cores = True
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        assert model_runner.sliding_window_size is None
        assert not model_runner.model_config.is_encoder_decoder 
        assert not self.skip_prefill
        assert not self.is_multimodal
        assert kv_indptr_buf is None
        assert kv_last_page_len_buf is None
        self.num_wrappers = 2
        self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
        ):
            global_config.flashinfer_workspace_size = 512 * 1024 * 1024

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                global_config.flashinfer_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_workspace_buffer
        max_bs = model_runner.req_to_token_pool.size
        
        self.num_qo_heads = model_runner.model_config.num_attention_heads // get_attention_tp_size()
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(get_attention_tp_size())
        self.group_size = self.num_qo_heads // self.num_kv_heads
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype

        assert self.q_data_type == torch.bfloat16
        assert self.data_type == torch.bfloat16
        
        # Assign key configuration and parameters
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.page_size = model_runner.server_args.page_size
        self.layers_skip = model_runner.server_args.vortex_layers_skip
        
        # ===========================
        # Decode cu seqlens q buffers
        # ===========================
        self.cu_seqlens_q_decode = torch.arange(
                    0, max_bs * self.num_kv_heads + 1, dtype=torch.int32, device=model_runner.device
                )
        
        
        # ===========================
        # Decode expanded cache_seqlens
        # ===========================
        self.cache_seqlens = [
            torch.arange(
                    0, max_bs * self.num_kv_heads + 1, dtype=torch.int32, device=model_runner.device
                ),
            torch.arange(
                    0, max_bs * self.num_kv_heads + 1, dtype=torch.int32, device=model_runner.device
                ),
        ]

        
        # ===========================
        # Decode KV-indptr buffers
        # ===========================

        self.kv_indptr_decode = [
            torch.zeros(
                (max_bs * self.num_kv_heads + 1,),
                dtype=torch.int32,
                device=model_runner.device
            ),
            torch.zeros(
                (max_bs * self.num_kv_heads + 1,),
                dtype=torch.int32,
                device=model_runner.device
            ),
        ]

        # ===========================
        # Page Table (prefill)
        # ===========================
        self.page_table_prefill = torch.zeros(
            (max_bs * self.num_kv_heads, (model_runner.model_config.context_len + self.page_size - 1) // self.page_size),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # KV indices (decode)
        # ===========================

        self.kv_indices_decode = [
            torch.zeros(
                (
                    (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1)
                    // self.page_size,
                ),
                dtype=torch.int32,
                device=model_runner.device
            ),
            torch.zeros(
                (
                    (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1)
                    // self.page_size,
                ),
                dtype=torch.int32,
                device=model_runner.device
            ),
        ]
        
        self.page_table_decode = [
            torch.zeros(
                (max_bs * self.num_kv_heads, (model_runner.model_config.context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=model_runner.device
            ),
            torch.zeros(
                (max_bs * self.num_kv_heads, (model_runner.model_config.context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=model_runner.device
            )
        ]

        # ===========================
        # KV last page length tracking
        # ===========================

        self.kv_last_page_len_decode = torch.ones(
            (max_bs * self.num_kv_heads,),
            dtype=torch.int32,
            device=model_runner.device
        )

       
        # ===========================
        # Batch table (token-level mapping)
        # ===========================

        self.batch_table = torch.zeros(
            (model_runner.server_args.max_prefill_tokens,),
            dtype=torch.uint16,
            device=model_runner.device
        )

        
        
        self.decode_wrappers = [
            BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                ),
            BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                ),
        ]
        
        self.sparse_attention = model_runner.sparse_attention
        self.ctx = vortex_torch.indexer.Context()
        self._initialize_graph(model_runner)
        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata, FlashAttentionMetadata] = None
        self.decode_cuda_graph_metadata: Dict[int, List[BatchDecodeWithPagedKVCacheWrapper]] = {}
        self.plan_graph: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]]
    

    def _initialize_graph(self, model_runner: "ModelRunner") -> None:
        """
        Initialize execution context and warm up kernels/graphs with minimal dummy tensors.

        Expectations:
            - self.head_dim: int > 0
            - self.ctx: provides create / assert_created / profile / summary / execute
            - self.sparse_attention.forward_indexer is callable
            - model_runner.device is a valid torch.device
        """
        # ---- Basic validations ----
        if getattr(self, "ctx", None) is None:
            raise RuntimeError("`self.ctx` is not set. Please construct/inject a context before initialize().")

        if not hasattr(self, "head_dim") or not isinstance(self.head_dim, int) or self.head_dim <= 0:
            raise AttributeError("`self.head_dim` must be a positive integer.")

        device: Optional[torch.device] = getattr(model_runner, "device", None)
        if device is None:
            raise AttributeError("`model_runner.device` is required but missing.")

        indexer = getattr(getattr(self, "sparse_attention", None), "forward_indexer", None)
        if indexer is None or not callable(indexer):
            raise AttributeError("`self.sparse_attention.forward_indexer` is missing or not callable.")

        # ---- Context lifecycle ----
        self.ctx.create(self, model_runner)
        self.ctx.assert_created()
        self.ctx.profile()  # enter 'profile' mode during warm-up

        # ---- Minimal warm-up tensors (placeholders only) ----
        dtype = torch.bfloat16

        try:
            with torch.no_grad():
                # Dummy placeholders: used only for kernel / graph warm-up
                q_dummy = as_vtensor(torch.empty((1, self.group_size, self.head_dim), device=device, dtype=dtype), FORMAT.BATCHED)
                o_dummy = as_vtensor(torch.empty((0, 1, 1), device=device, dtype=dtype), FORMAT.RAGGED)
                cache_meta_info = self.sparse_attention.get_cache_meta_info(self.page_size, self.head_dim)
                
                cache_dummy = {
                        cache_name:  as_vtensor(torch.zeros(
                                (0, cache_shape[0], cache_shape[1]),
                                dtype=dtype,
                                device=device,
                            ), FORMAT.PAGED)
                        
                        for (cache_name, cache_shape) in cache_meta_info.items()
                    }
                
                indexer(q_dummy, o_dummy, cache_dummy, ctx=self.ctx)


        except Exception:
            raise

        
        self.ctx.summary()
        self.ctx.execute()



    
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        
        
        assert not forward_batch.forward_mode.is_draft_extend()
        assert not forward_batch.forward_mode.is_target_verify()
        
        if forward_batch.forward_mode.is_decode_or_idle():
            
            metadata = FlashAttentionMetadata()
            metadata.eff_bs = forward_batch.batch_size * self.num_kv_heads
            vortex_torch.indexer.utils_sglang.plan_decode_fa3(
                cached_seq_lens=forward_batch.seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                dense_page_table=self.page_table_decode[0],
                dense_cache_seqlens=self.cache_seqlens[0],
                sparse_page_table=self.page_table_decode[1],
                sparse_cache_seqlens=self.cache_seqlens[1],
                ctx=self.ctx
            )
            
            self.forward_metadata = metadata
            
        elif forward_batch.forward_mode.is_extend():
            
            metadata = FlashAttentionMetadata()
            metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32).repeat_interleave(repeats=self.num_kv_heads, dim=0)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.unexpand_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
            
            
            if (
                any(forward_batch.extend_prefix_lens_cpu)
            ):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(
                        extend_seq_lens.repeat_interleave(repeats=self.num_kv_heads, dim=0), 
                        dim=0, dtype=torch.int32), 
                        (1, 0)
                    )
                metadata.unexpand_cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(
                        extend_seq_lens, 
                        dim=0, dtype=torch.int32), 
                        (1, 0)
                    )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k
                metadata.unexpand_cu_seqlens_q = metadata.unexpand_cu_seqlens_k
            

            vortex_torch.indexer.utils_sglang.plan_prefill_fa3(
                metadata.cache_seqlens_int32,
                metadata.cu_seqlens_q,
                self.req_to_token,
                forward_batch.req_pool_indices,
                self.page_table_prefill,
                self.batch_table,
                self.page_size,
                self.num_kv_heads
            )
            metadata.eff_bs = forward_batch.batch_size * self.num_kv_heads
            self.forward_metadata = metadata
           

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        pass
    
    
    def capture_plan_graph(
        self, 
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        bs: int):
        
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):  
        assert bs == num_tokens
        
        if forward_mode.is_decode_or_idle():
            decode_wrappers = [
                BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.kv_indptr_decode[0][:bs*self.num_kv_heads + 1],
                        paged_kv_indices_buffer=self.kv_indices_decode[0],
                        paged_kv_last_page_len_buffer=self.kv_last_page_len_decode[
                            :bs*self.num_kv_heads
                        ],
                    ),
                
                BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.kv_indptr_decode[1][:bs*self.num_kv_heads + 1],
                        paged_kv_indices_buffer=self.kv_indices_decode[1],
                        paged_kv_last_page_len_buffer=self.kv_last_page_len_decode[
                            :bs*self.num_kv_heads
                        ],
                    ),
                
            ]

            vortex_torch.indexer.utils_sglang.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )
            
            decode_wrappers[0].plan(
                indptr=self.kv_indptr_decode[0][:bs*self.num_kv_heads+1],
                indices=self.kv_indices_decode[0],
                last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            decode_wrappers[1].plan(
                indptr=self.kv_indptr_decode[1][:bs*self.num_kv_heads+1],
                indices=self.kv_indices_decode[1],
                last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)             
        else:
            raise NotImplementedError
            

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        assert forward_mode.is_decode_or_idle()
        
        vortex_torch.indexer.utils_sglang.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )
        
        self.decode_cuda_graph_metadata[bs][0].plan(
            indptr=self.kv_indptr_decode[0][:bs*self.num_kv_heads+1],
            indices=self.kv_indices_decode[0],
            last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
            num_qo_heads=self.group_size,
            num_kv_heads=1,
            head_dim=self.head_dim,
            page_size=self.page_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
        )
        
        self.decode_cuda_graph_metadata[bs][1].plan(
            indptr=self.kv_indptr_decode[1][:bs*self.num_kv_heads+1],
            indices=self.kv_indices_decode[1],
            last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
            num_qo_heads=self.group_size,
            num_kv_heads=1,
            head_dim=self.head_dim,
            page_size=self.page_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        
        assert isinstance(forward_batch.token_to_kv_pool, VTXGraphCachePool)
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        metadata = self.forward_metadata
        cu_seqlens_q = metadata.cu_seqlens_q
        unexpand_cu_seqlens_q = metadata.unexpand_cu_seqlens_q
        cache_seqlens = metadata.cache_seqlens_int32
        max_seqlen_q = metadata.max_seq_len_q
        cu_seqlens_k = metadata.cu_seqlens_k
        eff_bs = metadata.eff_bs
        q = q.contiguous()
        q_t = vortex_torch.indexer.utils_sglang.chunkwise_nh2hn_transpose(
                q.view(-1, self.num_qo_heads, self.head_dim),
                unexpand_cu_seqlens_q,
                self.batch_table,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim
            )
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(-1, self.page_size, 1, self.head_dim)
        v_cache = v_cache.view(-1, self.page_size, 1, self.head_dim)
        
        o_t = flash_attn_with_kvcache(
                q_t,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=self.page_table_prefill[:eff_bs],
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=True,
                window_size=(-1, -1),
                softcap=layer.logit_cap,
                k_descale=None,
                v_descale=None,
                return_softmax_lse=False,
            )
        o = vortex_torch.indexer.utils_sglang.chunkwise_hn2nh_transpose_fa3(
            o_t,
            unexpand_cu_seqlens_q,
            self.batch_table,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim
        )
        
        
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """
        Decode-time forward pass with optional sparse attention.
        Expects KV to be sourced from token_to_kv_pool; can also save new KV.
        """

        # Sanity checks and setup
        assert isinstance(forward_batch.token_to_kv_pool, VTXGraphCachePool)
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc
        metadata = self.forward_metadata
        # Optionally write incoming K/V to decode cache
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Read Cache from memory pool
        cache = forward_batch.token_to_kv_pool.get_cache(layer.layer_id)
        
        cache_k = cache["k"].view(-1, self.page_size, 1, self.head_dim)
        cache_v = cache["v"].view(-1, self.page_size, 1, self.head_dim)
        
        # Decide whether to use sparsity on this layer
        use_sparsity = (layer.layer_id not in self.layers_skip)
        # Dense attention path
        
        o = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, self.group_size, layer.head_dim),
                k_cache=cache_k,
                v_cache=cache_v,
                page_table=self.page_table_decode[0][:metadata.eff_bs],
                cache_seqlens=self.cache_seqlens[0][:metadata.eff_bs],
                cu_seqlens_q=self.cu_seqlens_q_decode[:metadata.eff_bs + 1],
                cu_seqlens_k_new=self.kv_indptr_decode[0][:metadata.eff_bs + 1],
                max_seqlen_q=1,
                softmax_scale=layer.scaling,
                causal=True,
                window_size=(-1, -1),
                softcap=layer.logit_cap,
                k_descale=None,
                v_descale=None,
                return_softmax_lse=False,
                )
        
       

        # Restore to merged head dimension
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)