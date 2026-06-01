#include "register.h"
#include <cub/cub.cuh>

struct PairSum {
  __device__ __forceinline__ int2 operator()(const int2& a, const int2& b) const {
    return make_int2(a.x + b.x, a.y + b.y);
  }
};

__launch_bounds__(1024, 1)
__global__ void Sgl_Decode_Plan_Indptr_V2_Kernel(
const int*  __restrict__ cached_seq_lens,
int*  __restrict__ dense_kv_indptr,
int*  __restrict__ sparse_kv_indptr,
const int batch_size,
const int num_kv_heads,
const int page_size,
const int block_size,
const int topk_val,
const float topk_ratio,
const int block_reserved_bos,
const int block_reserved_eos
){

    const int tx = threadIdx.x;
    const int cached_seq_len = (tx < batch_size) ? cached_seq_lens[tx] : 0;
    const int cached_block_len = (cached_seq_len + block_size - 1) / block_size;

    const int static_kv_budget = topk_val + block_reserved_bos + block_reserved_eos;
    const int dynamic_kv_budget = int(cached_block_len * topk_ratio);
    const int kv_budget = max(static_kv_budget, dynamic_kv_budget);

    using BlockScanInt2 = cub::BlockScan<int2, 1024>;

    __shared__ union {
        typename BlockScanInt2::TempStorage scan_int2;
    } temp;

   
    const int block_cnt = (tx < batch_size) ? cached_block_len : 0;
    const int sparse_cnt = (tx < batch_size) ? min(kv_budget, cached_block_len) : 0;
    
    int2 in  = make_int2(block_cnt, sparse_cnt);
    int2 out;
    BlockScanInt2(temp.scan_int2).InclusiveScan(in, out, PairSum{});

    const int dense_cumsum  = out.x;
    const int sparse_cumsum = out.y;

    if (tx < batch_size){
        #pragma unroll
        for (int i = 0; i < num_kv_heads; ++i){
            dense_kv_indptr[num_kv_heads * (tx + 1) - i] = dense_cumsum * num_kv_heads
                - i * cached_block_len;
            
            sparse_kv_indptr[num_kv_heads * (tx + 1) - i] = sparse_cumsum * num_kv_heads
                - i * min(kv_budget, cached_block_len);
        }

    }

    if(tx == 0){
        dense_kv_indptr[0] = 0;
        sparse_kv_indptr[0] = 0;
    }

};



template <int ITEM_PER_THREAD>
__launch_bounds__(1024, 1)
__global__ void Sgl_Decode_Plan_Workload_V2_Kernel(
const int*  __restrict__ dense_kv_indptr,
const int*  __restrict__ sparse_kv_indptr,
int*  __restrict__ winfo_q_indices,
int*  __restrict__ winfo_kv_offsets,
int*  __restrict__ winfo_kv_lens,
int*  __restrict__ winfo_num_workloads,
int*  __restrict__ winfo_chunk_size,
const int workload_chunk_size,
const int eff_batch_size,
// const int topk_val,
const int block_reserved_bos,
const int block_reserved_eos
){  

    const int tx = threadIdx.x;
    using BlockScanInt = cub::BlockScan<int, 1024>;
    __shared__ union {
        typename BlockScanInt::TempStorage scan_int;
    } temp;


    uint16_t block_count[ITEM_PER_THREAD];
    int chunked_block_count_prefix_sum[ITEM_PER_THREAD + 1];
    int tx_offset = tx * ITEM_PER_THREAD;

    chunked_block_count_prefix_sum[0] = 0;

    #pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i){


        // int16_t w = ((tx_offset + i) < eff_batch_size) ? 
        //     (dense_kv_indptr[tx_offset+i+1] - dense_kv_indptr[tx_offset+i] - block_reserved_eos): 0;
    
        // block_count[i] = (w > topk_val + block_reserved_bos) ? w : 0;
        int16_t w = 0;
        if((tx_offset + i) < eff_batch_size){
            int16_t dense_seqlen_i = dense_kv_indptr[tx_offset+i+1] - dense_kv_indptr[tx_offset+i];
            int16_t sparse_seqlen_i = sparse_kv_indptr[tx_offset+i+1] - sparse_kv_indptr[tx_offset+i];
            if(dense_seqlen_i > sparse_seqlen_i){
                w = dense_seqlen_i - block_reserved_eos;
            } else {
                w = 0;
            }
        }
        block_count[i] = w;
        chunked_block_count_prefix_sum[i + 1] =  int((block_count[i] + workload_chunk_size - 1) / workload_chunk_size);
    }

    BlockScanInt(temp.scan_int).InclusiveSum(chunked_block_count_prefix_sum, chunked_block_count_prefix_sum);
    
    if (tx == 1023){
        *winfo_num_workloads = chunked_block_count_prefix_sum[ITEM_PER_THREAD];
        *winfo_chunk_size = workload_chunk_size;
    }

    
    for (int i = 0; i < ITEM_PER_THREAD; ++i){

        if((tx_offset + i) < eff_batch_size){
        const int start = chunked_block_count_prefix_sum[i];
        const int end = chunked_block_count_prefix_sum[i+1];
        int last_len = int(block_count[i] % workload_chunk_size);
        if (last_len == 0) last_len = workload_chunk_size;
        for (int j = start; j < end; ++j){
                winfo_q_indices[j] = tx_offset + i;
                winfo_kv_lens[j] = (j!=end-1)?(workload_chunk_size):(last_len);
                winfo_kv_offsets[j] = dense_kv_indptr[tx_offset + i] + (j - start) * workload_chunk_size;
        }

        }

    }

};

__global__ void Sgl_Decode_Plan_Indices_V2_Kernel(
const int*   __restrict__ req_to_token,
const long*  __restrict__ req_indices,
const int*   __restrict__ cache_seq_lens,
const int*   __restrict__ dense_kv_indptr,
const int*   __restrict__ sparse_kv_indptr,
int*         __restrict__ dense_kv_indices,
int*         __restrict__ sparse_kv_indices,
int*         __restrict__ kv_last_block_len,
const int    req_to_token_stride,
const int    page_size,
const int    block_size,
const int    num_kv_heads,
const int    block_reserved_bos,
const int    block_reserved_eos
){

    const int nblk = blockDim.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int* token_indices = req_to_token + req_indices[bx] * req_to_token_stride;
    const int kv_len = cache_seq_lens[bx];
    const int block_len = (kv_len + block_size - 1) / block_size;
    const int last_len = kv_len % block_size;
    const int num_blocks_per_page = page_size / block_size;
    if (tx == 0) {
            kv_last_block_len[bx * num_kv_heads + by] = (last_len == 0) ? block_size : last_len;
    }

    int* dense_output = dense_kv_indices + dense_kv_indptr[bx * num_kv_heads + by];
    int* sparse_output = sparse_kv_indices + sparse_kv_indptr[bx * num_kv_heads + by];
    const int kv_budget = sparse_kv_indptr[bx * num_kv_heads + by + 1] - sparse_kv_indptr[bx * num_kv_heads + by];
    // const int kv_budget = topk_val + block_reserved_bos + block_reserved_eos;
    int pos = tx;
    while(pos < block_len){
            int data = token_indices[pos * block_size];
            int page_id = (data / page_size) * (num_kv_heads) + by;
            dense_output[pos] = page_id * num_blocks_per_page + (data % page_size) / block_size;
            pos += nblk;
    }

    if(block_len <= kv_budget){
        int pos = tx;
        while(pos < block_len){
            sparse_output[pos] = dense_output[pos];
            pos += nblk;
        }
    }else{
        int pos = tx;
        while(pos < block_reserved_bos){
            sparse_output[pos] = dense_output[pos];
            pos += nblk;
        }

        pos = tx;
        while(pos < block_reserved_eos){
            int data = token_indices[(block_len-pos-1) * block_size];
            int page_id = (data / page_size) * (num_kv_heads) + by;
            sparse_output[kv_budget-pos-1] = page_id * num_blocks_per_page + (data % page_size) / block_size;
            pos += nblk;
        }

    }
}


void sglang_plan_decode_v2(
const at::Tensor&   cached_seq_lens,
at::Tensor&         dense_kv_indptr,
at::Tensor&         dense_kv_indices,
at::Tensor&         sparse_kv_indptr,
at::Tensor&         sparse_kv_indices,
at::Tensor&         kv_last_page_len,
const at::Tensor&   req_to_token,
const at::Tensor&   req_indices,
at::Tensor&         winfo_q_indices,
at::Tensor&         winfo_kv_offsets,
at::Tensor&         winfo_kv_lens,
at::Tensor&         winfo_num_workloads,
at::Tensor&         winfo_chunk_size,
const int64_t       page_size,
const int64_t       block_size,
const int64_t       num_kv_heads,
const int64_t       topk_val,
const float         topk_ratio,
const int64_t       block_reserved_bos,
const int64_t       block_reserved_eos,
const int64_t       workload_chunk_size
) {

#ifdef VORTEX_DEBUG
    TORCH_CHECK(req_to_token.dtype() == torch::kInt32, "req_to_token must be int32");
    TORCH_CHECK(req_indices.dtype() == torch::kInt64, "req_indices must be int64");
    TORCH_CHECK(cached_seq_lens.dtype() == torch::kInt32, "cached_seq_lens must be int32");
    TORCH_CHECK(dense_kv_indptr.dtype() == torch::kInt32);
    TORCH_CHECK(dense_kv_indices.dtype() == torch::kInt32);
    TORCH_CHECK(sparse_kv_indptr.dtype() == torch::kInt32);
    TORCH_CHECK(sparse_kv_indices.dtype() == torch::kInt32);
    TORCH_CHECK(kv_last_page_len.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_q_indices.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_kv_offsets.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_kv_lens.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_num_workloads.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_chunk_size.dtype() == torch::kInt32);
    TORCH_CHECK(page_size >= 1);
    TORCH_CHECK(block_size >= 1);
    TORCH_CHECK(page_size % block_size == 0, "page_size must be divisible by block_size");
    TORCH_CHECK(num_kv_heads >= 1);
    TORCH_CHECK(topk_val >= 1);
    TORCH_CHECK(topk_ratio >= 0.0f && topk_ratio <= 1.0f);
    TORCH_CHECK(block_reserved_bos >= 0);
    TORCH_CHECK(block_reserved_eos >= 1);
    TORCH_CHECK(min_chunk_size >= 1);
    TORCH_CHECK(max_chunk_size > min_chunk_size);
#endif

    const int batch_size = cached_seq_lens.size(0);
    const int eff_batch_size = batch_size * num_kv_heads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    Sgl_Decode_Plan_Indptr_V2_Kernel<<<1, 1024, 0, stream>>>(
        cached_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        batch_size,
        num_kv_heads,
        page_size,
        block_size,
        topk_val,
        topk_ratio,
        block_reserved_bos,
        block_reserved_eos
    );

    if (eff_batch_size <= 1024){
    Sgl_Decode_Plan_Workload_V2_Kernel<1><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        workload_chunk_size,
        eff_batch_size,
        block_reserved_bos,
        block_reserved_eos
    );
    } else if (eff_batch_size <= 2048){
    Sgl_Decode_Plan_Workload_V2_Kernel<2><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        workload_chunk_size,
        eff_batch_size,
        block_reserved_bos,
        block_reserved_eos
    );
    } else if (eff_batch_size <= 4096){
    Sgl_Decode_Plan_Workload_V2_Kernel<4><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        workload_chunk_size,
        eff_batch_size,
        block_reserved_bos,
        block_reserved_eos
    );
    }  else if (eff_batch_size <= 8192){
    Sgl_Decode_Plan_Workload_V2_Kernel<8><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        workload_chunk_size,
        eff_batch_size,
        block_reserved_bos,
        block_reserved_eos
    );
    }  

    // Fill in indices for dense and sparse (eos and bos)

    const int req_to_token_stride = req_to_token.size(1);
    dim3 nblks(batch_size, num_kv_heads);
    dim3 nthrs(512);
    Sgl_Decode_Plan_Indices_V2_Kernel<<<nblks, nthrs, 0, stream>>>(
        req_to_token.data_ptr<int>(),
        req_indices.data_ptr<long>(),
        cached_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        kv_last_page_len.data_ptr<int>(),
        req_to_token_stride,
        page_size,
        block_size,
        num_kv_heads,
        block_reserved_bos,
        block_reserved_eos
    );
}