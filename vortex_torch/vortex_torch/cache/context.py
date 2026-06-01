from __future__ import annotations
from typing import Any
from ..utils import UNSET, Mode
from ..abs import ContextBase


class Context(ContextBase):
    """
    Mutable, single-instance context; populate later via .create(...).
    """
    
    __slots__ = ContextBase.__slots__ + (
        
        #page & block infomation
        "max_new_tokens_per_batch", "page_size", "total_num_pages", "block_size", "num_blocks_per_page", "total_num_blocks",
        
        #model infomation
        "head_dim", "head_num",
        
        # auxilary memory in graph
        "_aux_total_bytes",
        
        "_aux_total_flops"
    )


    def __init__(self) -> None:
        # Start as an empty shell (no big allocations).
        for name in self.__slots__:
            if name == "_created":
                object.__setattr__(self, name, False)
            elif name == "name":
                object.__setattr__(self, name, "Cache")
            elif name == "_aux_total_bytes":
                object.__setattr__(self, name, 0)  # start from 0 bytes
            elif name == "_aux_total_flops":
                object.__setattr__(self, name, 0)  # start from 0 flops
            elif name == "mode":
                object.__setattr__(self, name, Mode.profile) 
            else:
                object.__setattr__(self, name, UNSET)
    
     
    def create(self, parent: Any, model_runner: Any, *, overwrite: bool = False) -> "Context":
        """
        Populate this instance once (no locking). Set overwrite=True to allow re-init.
        NOTE: Without locking, concurrent callers may race; call from a single thread.
        """
        if self._created and not overwrite:
            raise RuntimeError("Context.create() already called; pass overwrite=True to reinitialize.")

        sa = model_runner.server_args
        self.page_size = parent.page_size
        self.block_size = parent.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size for block-sparse attention"
        self.num_blocks_per_page = self.page_size // self.block_size
        self.total_num_blocks = self.num_blocks_per_page * parent.num_pages
        self.total_num_pages = parent.num_pages
        self.max_new_tokens_per_batch = max(sa.max_prefill_tokens, model_runner.req_to_token_pool.size)
        self.head_num = parent.head_num
        self.head_dim = parent.head_dim
        self._created = True
        return self
    

