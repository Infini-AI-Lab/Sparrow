"""
Copyright 2025 Zhuoming Chen
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import logging
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
from contextlib import nullcontext
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.utils import (
    debug_timing,
    is_cuda
)

import vortex_torch
from vortex_torch.abs import as_vtensor, FORMAT
logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()

"""
Sparse Attention Memory pool.

In addition to Memory Pool in the original SGLang
We 
1) maintain auxilary cache tensor objects for every page.
2) internally treat each KV head as a request (as they may have different sparse patterns), 
then we interpret external auguments to the physical address
"""

class VTXGraphCachePool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        sparse_attention: vortex_torch.flow.vFlow,
        model_runner,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim

        # for disagg with nvlink
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.num_pages = ((self.size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size + 1
        
        self.sparse_attention = sparse_attention
        self.ctx = vortex_torch.cache.Context()
        self.block_size = model_runner.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size for block-sparse attention"
        self.num_blocks_per_page = self.page_size // self.block_size
        self._create_buffers()
        self._initialize_graph(model_runner)
        self.layer_transfer_counter = None
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None
        self.layers_skip = model_runner.server_args.vortex_layers_skip
        cache_size = self.get_cache_size_bytes()
        
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, Cache size: {cache_size / GB:.2f} GB"
        )
        
        self.mem_usage = cache_size / GB
        assert self.dtype in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn], f"Unsupported dtype {self.dtype} for KV cache"
        assert self.store_dtype in [torch.bfloat16, torch.uint8], f"Unsupported store dtype {self.store_dtype} for KV cache"
        if self.dtype == torch.bfloat16:
            self.set_kv_buffer_func = vortex_torch.cache.set_kv_buffer_launcher
        elif self.dtype == torch.float8_e4m3fn:
            self.set_kv_buffer_func = vortex_torch.cache.set_kv_buffer_fp8_e4m3_launcher
        elif self.dtype == torch.float8_e5m2:
            self.set_kv_buffer_func = vortex_torch.cache.set_kv_buffer_fp8_e5m2_launcher
        else:
            raise ValueError(f"Unsupported dtype {self.dtype} for KV cache")
        
    def _initialize_graph(self, model_runner) -> None:
        
        self.ctx.create(self, model_runner)
        self.ctx.profile()
        try:
            with torch.no_grad():
                loc_dummy = torch.empty((0,), dtype=torch.int64, device=self.device)
                cache_dummy = {
                        cache_name:  as_vtensor(torch.zeros(
                                (0 * self.num_blocks_per_page, cache_shape[0], cache_shape[1]),
                                dtype=cache_dtype,
                                device=self.device,
                            ), FORMAT.PAGED)
                        
                        for (cache_name, (cache_shape, cache_dtype)) in self.cache_meta_info.items()
                }
                self.sparse_attention.forward_cache(cache=cache_dummy, loc=loc_dummy, ctx=self.ctx)      
        except Exception:
            raise
        
        self.ctx.summary()
        self.ctx.execute()


    def _create_buffers(self):
        
        self.cache_meta_info = self.sparse_attention.get_cache_meta_info()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):  
                self.cache = [
                    {
                        cache_name:  torch.zeros(
                                (self.num_pages * self.num_blocks_per_page, cache_shape[0], cache_shape[1]),
                                dtype=cache_dtype,
                                device=self.device,
                            )
                        
                        for (cache_name, (cache_shape, cache_dtype)) in self.cache_meta_info.items()
                    }
                    
                    for _ in range(self.layer_num)
                ]
        
    def _clear_buffers(self):
        del self.cache
       

    def get_cache_size_bytes(self) -> int:
        """
        Return total bytes occupied by all tensors in `self.cache`.
        Works even if some entries are not tensors.
        """
        total_bytes = 0

        for layer_cache in self.cache:
            if not isinstance(layer_cache, dict):
                # Be tolerant to unexpected structures
                continue

            for t in layer_cache.values():
                if not torch.is_tensor(t):
                    continue

                # Prefer accurate allocated size if available (includes padding/strides)
                try:
                    total_bytes += int(t.untyped_storage().nbytes())
                except AttributeError:
                    # Fallback: logical size in bytes
                    total_bytes += int(t.element_size() * t.numel())

        return total_bytes
    
    def get_kv_size_bytes(self):
        
        raise NotImplementedError
    
    # for disagg
    def get_contiguous_buf_infos(self):
        
        raise NotImplementedError

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_cpu_copy(self, indices):
        
        raise NotImplementedError

    def load_cpu_copy(self, kv_cache_cpu, indices):
        
        raise NotImplementedError

    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        raise NotImplementedError


    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
       raise NotImplementedError

    def transfer_per_layer(self, indices, flat_data, layer_id):
        
        raise NotImplementedError


    def get_key_buffer(self, layer_id: int):
        
        return self.cache[layer_id - self.start_layer]["k"]

    def get_value_buffer(self, layer_id: int):
        
        return self.cache[layer_id - self.start_layer]["v"]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        return self.cache[layer_id - self.start_layer]["k"], self.cache[layer_id - self.start_layer]["v"]

        
    def get_cache(self, layer_id: int)->Dict[str, torch.Tensor]:
        
        return self.cache[layer_id - self.start_layer]

        
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        
        assert layer_id_override is None
        assert k_scale is None
        assert v_scale is None
        assert loc.dtype == torch.int64
        
        layer_id = layer.layer_id
        
        self.set_kv_buffer_func(
            self.cache[layer_id - self.start_layer]["k"],
            self.cache[layer_id - self.start_layer]["v"],
            cache_k.contiguous(),
            cache_v.contiguous(),
            loc,
            self.page_size
        )
        if layer_id in self.layers_skip:
            return
        self.sparse_attention.forward_cache(self.cache[layer_id - self.start_layer], loc, ctx=self.ctx)
        
    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        
        raise NotImplementedError