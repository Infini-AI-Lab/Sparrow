from __future__ import annotations
from typing import Any, Final, Union
import torch
from ..abs import ContextBase
from ..utils import UNSET, Mode
import uuid

class Context(ContextBase):
    """
    Mutable, single-instance context; populate later via .create(...).
    """
    
    __slots__ =  ContextBase.__slots__ + (
        # indices / indptr
        "dense_kv_indices", "sparse_kv_indices", "dense_kv_indptr", "sparse_kv_indptr", "kv_last_page_len", "batch_size",
        # winfo
        "winfo_q_indices", "winfo_kv_offsets", "winfo_kv_lens", "winfo_num_workloads", "winfo_chunk_size", "max_num_workloads",
        # chunk limits
        "workload_chunk_size",
        # head / shape
        "group_size", "num_kv_heads", "num_qo_heads", "head_dim",
        # hardware / paging
        "num_sms", "page_size", "max_num_pages", "max_num_pages_per_request", "block_size", "max_num_blocks", "max_num_blocks_per_request", "num_blocks_per_page", "num_pages_per_workload",
        # misc
        "indexer_dtype", "topk_val", "topk_ratio", "block_reserved_bos", "block_reserved_eos", 
        
        # auxilary memory in graph
        "_aux_total_bytes",
        
        # auxilary flops in graph
        "_aux_total_flops",

        "tensor_list", "op_list", "output_tensor_to_op_list", "op_to_input_tensor_list", "op_to_output_tensor_list",

        "sparse_attention_name", "impl_backend", "tensor_id_to_tensor_name_map", "compilation_header_lines", "auxilary_func_def_lines",

        "compilation_cache_dir", "model_path",
        )
    
    # --- index tensors ---
    dense_kv_indices: torch.Tensor  #: Dense KV index tensor for mapping keys/values.
    sparse_kv_indices: torch.Tensor  #: Sparse KV index tensor for irregular KV layout.
    dense_kv_indptr: torch.Tensor    #: CSR-style indptr for dense KV segments.
    sparse_kv_indptr: torch.Tensor   #: CSR-style indptr for sparse KV segments.
    kv_last_page_len: int            #: Length of the last KV page.
    batch_size: int                  #: Active batch size.

    # --- workload info (winfo) ---
    winfo_q_indices: torch.Tensor    #: Query indices used in workload scheduling.
    winfo_kv_offsets: torch.Tensor   #: KV offsets per workload.
    winfo_kv_lens: torch.Tensor      #: KV lengths per workload.
    winfo_num_workloads: int         #: Number of workloads in the current batch.
    winfo_chunk_size: int            #: Chunk size for workload partitioning.
    max_num_workloads: int           #: Maximum number of workloads allowed.

    # --- chunk limits ---
    workload_chunk_size: int              #: allowed chunk size.

    # --- head / shape configuration ---
    group_size: int                  #: Group size for grouped attention.
    num_kv_heads: int                #: Number of KV heads.
    num_qo_heads: int                #: Number of query/output heads.
    head_dim: int                    #: Dimension per attention head.

    # --- hardware / paging ---
    num_sms: int                     #: Number of streaming multiprocessors (SMs).
    page_size: int                   #: Page size used for memory paging.
    block_size: int                   #: Page size used for memory paging.
    max_num_pages: int               #: Total available pages.
    max_num_pages_per_request: int   #: Page limit per individual request.
    max_num_blocks: int               #: Total available pages.
    max_num_blocks_per_request: int   #: Page limit per individual request.
    num_blocks_per_page: int        #: Number of blocks contained in a single page.
    num_pages_per_workload: int      #: Number of pages processed per workload (derived from chunk size).

    # --- miscellaneous ---
    indexer_dtype: torch.dtype       #: Dtype used by indexer operations.
    topk_val: int                    #: Top-K value used in pruning or selection.
    topk_ratio: float                #: Top-K ratio used in pruning or selection.
    block_reserved_bos: int           #: Reserved page count for BOS (begin-of-sequence).
    block_reserved_eos: int           #: Reserved page count for EOS (end-of-sequence).

    # --- auxiliary ---
    _aux_total_bytes: int            #: Accumulated auxiliary memory in bytes.
    _aux_total_flops: int            #: Accumulated auxiliary flops.
    tensor_list: list                #: List of tensors used in the graph.
    op_list: list                    #: List of operations in the graph.
    output_tensor_to_op_list: list    #: Mapping from output tensors to their producing operations.
    op_to_input_tensor_list: list     #: Mapping from operations to their input tensors.
    op_to_output_tensor_list: list    #: Mapping from operations to their output tensors.
    sparse_attention_name: str          #: Name of the sparse attention implementation to use.
    impl_backend: str       #: Implementation backend to use for code generation.
    tensor_id_to_tensor_name_map: dict #: Mapping from tensor IDs to human-readable names for debugging.
    compilation_header_lines: list       #: Header string to prepend to generated code during compilation.
    auxilary_func_def_lines: list       #: List of auxiliary function definitions to include in the generated code.
    compilation_cache_dir: str          #: Directory path for caching compiled kernels.
    model_path: str                     #: Model identifier used to launch the engine.
    def __init__(self) -> None:
        # Start as an empty shell (no big allocations).
        for name in self.__slots__:
            if name == "_created":
                object.__setattr__(self, name, False)
            elif name == "name":
                object.__setattr__(self, name, "Indexer")
            elif name == "_aux_total_bytes":
                object.__setattr__(self, name, 0)  # start from 0 bytes
            elif name == "_aux_total_flops":
                object.__setattr__(self, name, 0)  # start from 0 flops
            elif name == "batch_size":
                object.__setattr__(self, name, 0)
            elif name == "mode":
                object.__setattr__(self, name, Mode.profile) 
            else:
                object.__setattr__(self, name, UNSET)

    
    def set_batch_size(self, n: int) -> None:
        
        self.batch_size = n
        
    
    def create(self, parent: Any, model_runner: Any, *, overwrite: bool = False) -> "Context":
        """
        Populate this instance once (no locking). Set overwrite=True to allow re-init.
        NOTE: Without locking, concurrent callers may race; call from a single thread.
        """
        if self._created and not overwrite:
            raise RuntimeError("Context.create() already called; pass overwrite=True to reinitialize.")

        sa = model_runner.server_args
        max_pages_per_req = (
            (model_runner.model_config.context_len + sa.page_size - 1) // sa.page_size
            if sa.vortex_max_seq_lens < 0
            else (sa.vortex_max_seq_lens + sa.page_size - 1) // sa.page_size
        )
        max_bs = int(model_runner.req_to_token_pool.size)

        # Backend-known fields
        self.dense_kv_indices = parent.kv_indices_decode[0]
        self.sparse_kv_indices = parent.kv_indices_decode[1]
        self.dense_kv_indptr = parent.kv_indptr_decode[0]
        self.sparse_kv_indptr = parent.kv_indptr_decode[1]
        self.kv_last_page_len = parent.kv_last_page_len_decode

        self.workload_chunk_size = sa.vortex_workload_chunk_size

        self.group_size = parent.group_size
        self.num_kv_heads = parent.num_kv_heads
        self.num_qo_heads = parent.num_qo_heads
        self.head_dim = parent.head_dim

        self.num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.page_size = sa.page_size
        self.block_size = sa.vortex_block_size
        self.num_blocks_per_page = self.page_size // self.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size."
        assert self.workload_chunk_size % self.num_blocks_per_page == 0, "Workload chunk size must be a multiple of blocks per page."
        # Capacity model (adjust as needed)
        self.max_num_pages = max_pages_per_req * max_bs * self.num_kv_heads
        self.max_num_pages_per_request = max_pages_per_req
        self.max_num_blocks = self.max_num_pages * self.num_blocks_per_page
        self.max_num_blocks_per_request = self.max_num_pages_per_request * self.num_blocks_per_page
        self.num_pages_per_workload = self.workload_chunk_size // self.num_blocks_per_page
        self.topk_val = sa.vortex_topk_val
        self.topk_ratio = sa.vortex_topk_ratio
        dtype_str = getattr(sa, "vortex_indexer_dtype", "float32")
        if isinstance(dtype_str, str):
            self.indexer_dtype = getattr(torch, dtype_str, torch.float32)
        else:
            self.indexer_dtype = dtype_str
        
        self.block_reserved_bos = sa.vortex_block_reserved_bos
        self.block_reserved_eos = sa.vortex_block_reserved_eos

        self.max_num_workloads = (
            (self.max_num_blocks // max(1, self.workload_chunk_size)) + max_bs * self.num_kv_heads
        )

        device = getattr(model_runner, "device", "cpu")
        self.winfo_q_indices = torch.zeros((self.max_num_workloads,), dtype=torch.int32, device=device)
        self.winfo_kv_offsets = torch.zeros((self.max_num_workloads,), dtype=torch.int32, device=device)
        self.winfo_kv_lens = torch.zeros((self.max_num_workloads,), dtype=torch.int32, device=device)
        self.winfo_num_workloads = torch.zeros((1,), dtype=torch.int32, device=device)
        self.winfo_chunk_size = torch.zeros((1,), dtype=torch.int32, device=device)

        self.tensor_list = []
        self.op_list = []
        self.output_tensor_to_op_list = []
        self.op_to_input_tensor_list = []
        self.op_to_output_tensor_list = []
        self.tensor_id_to_tensor_name_map = {}
        self.compilation_header_lines = []
        self.auxilary_func_def_lines = []
        self.compilation_cache_dir = sa.vortex_compilation_cache_dir
        self.model_path = getattr(sa, "model_path", "")
        self.sparse_attention_name = parent.sparse_attention.__class__.__name__.lower() + f"_{uuid.uuid4().hex[:8]}"  # unique name for this attention instance 
        self.impl_backend = "triton"  # default to triton; can be overridden by user
        self._created = True
        return self



# Module-level singleton (part of the public package API)
ctx: Final[Context] = Context()

def get_ctx() -> Context:
    return ctx

__all__ = ["Context", "ctx", "get_ctx"]
