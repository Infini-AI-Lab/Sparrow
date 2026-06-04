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
from typing import List, Optional, Tuple, Union

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
from vortex import set_kv_buffer_launcher, update_landmark_launcher, FuseUpdateKVLandmark
logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()

"""
Sparse Attention Memory pool.

In addition to Memory Pool in the original SGLang
We 
1) maintain a landmark tensor for every page.
2) internally treat each KV head as a request (as they may have different sparse patterns), 
then we interpret external auguments to the physical address
"""

class VTXTokenToKVPool(KVCache):

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
        
        self._create_buffers()

        self.layer_transfer_counter = None
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None

        k_size, v_size = self.get_kv_size_bytes()
        landmark_size = self.get_landmark_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB, Landmark size: {landmark_size / GB:.2f} GB."
        )
        
        self.mem_usage = (k_size + v_size + landmark_size) / GB
        assert self.dtype == torch.bfloat16
        assert self.store_dtype == torch.bfloat16
        
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.k_buffer = [
                    torch.zeros(
                        (self.num_pages, self.page_size, 1, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (self.num_pages, self.page_size, 1, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                
                self.landmark_buffer = [
                    torch.zeros(
                        (self.num_pages, 1,  1, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer + self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.landmark_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes
    
    def get_landmark_size_bytes(self):
        assert hasattr(self, "landmark_buffer")
        landmark_size_bytes = 0
        for landmark_cache in self.landmark_buffer:
            landmark_size_bytes += np.prod(landmark_cache.shape) * landmark_cache.dtype.itemsize
        
        return landmark_size_bytes

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
        
        assert self.layer_transfer_counter is None
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        
        assert self.layer_transfer_counter is None
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer], self.v_buffer[layer_id - self.start_layer]

    def get_landmark_buffer(self, layer_id: int):
        
        return self.landmark_buffer[layer_id - self.start_layer]
    
    
    
    def set_kv_buffer_decode(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        
        
        layer_id = layer.layer_id
        
        FuseUpdateKVLandmark(
            cache_k.contiguous(),
            cache_v.contiguous(),
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            self.landmark_buffer[layer_id - self.start_layer],
            loc,
            self.page_size
        )
        
        
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
        assert cache_k.dtype == torch.bfloat16
        assert cache_v.dtype == torch.bfloat16
        assert loc.dtype == torch.int64
        
        layer_id = layer.layer_id
        
        set_kv_buffer_launcher(
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            cache_k.contiguous(),
            cache_v.contiguous(),
            loc,
            self.page_size
        )
        
        update_landmark_launcher(
            self.k_buffer[layer_id - self.start_layer],
            self.landmark_buffer[layer_id - self.start_layer],
            loc,
            self.page_size,
            self.head_num,
            self.head_dim
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        
        raise NotImplementedError