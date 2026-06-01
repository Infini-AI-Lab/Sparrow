#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int WARP_SIZE  = 32;
constexpr int NUM_WARPS  = 8;
constexpr int NUM_THREADS = WARP_SIZE * NUM_WARPS;    // blockDim = (32, 8, 1)
constexpr int BF16_PER_VEC = 8;                       // 16B / sizeof(bf16)

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

}  // namespace


template <int G, int D, int W, int NBLOCKS_PER_PAGE, int NPAGES_PER_WORKLOAD>
__device__ __forceinline__ void async_copy_x(
const __nv_bfloat16* __restrict__ x_ptr,
__nv_bfloat16*       __restrict__ x_shm_buffer,       // -> [G][D]
const int            workload_id,
const int*           __restrict__ winfo_x_indices_ptr,
const int            /*phase*/,
const int            /*bx*/,
const int            tx,
const int            ty,
const int            /*tz*/
){
    constexpr int TOTAL_VEC = (G * D) / BF16_PER_VEC;
    static_assert((G * D) % BF16_PER_VEC == 0, "G*D must be multiple of 8");

    const int tid   = ty * WARP_SIZE + tx;
    const int x_row = winfo_x_indices_ptr[workload_id];
    const __nv_bfloat16* src_base =
        x_ptr + static_cast<size_t>(x_row) * G * D;

    #pragma unroll
    for (int v = tid; v < TOTAL_VEC; v += NUM_THREADS) {
        cp_async_16(smem_addr(x_shm_buffer + v * BF16_PER_VEC),
                    src_base + v * BF16_PER_VEC);
    }
}


template <int G, int D, int W, int NBLOCKS_PER_PAGE, int NPAGES_PER_WORKLOAD>
__device__ __forceinline__ void async_copy_y(
const __nv_bfloat16* __restrict__ y_ptr,
__nv_bfloat16*       __restrict__ y_shm_buffer,       // -> [W][D]
const int            workload_id,
const int*           __restrict__ winfo_y_indices_ptr,
const int*           __restrict__ winfo_y_offsets_ptr,
const int*           __restrict__ winfo_y_lens_ptr,
const int            /*phase*/,
const int            /*bx*/,
const int            tx,
const int            ty,
const int            /*tz*/
){
    constexpr int PAGE_BF16 = NBLOCKS_PER_PAGE * D;
    constexpr int PAGE_VEC  = PAGE_BF16 / BF16_PER_VEC;
    constexpr int TOTAL_VEC = NPAGES_PER_WORKLOAD * PAGE_VEC;
    static_assert(PAGE_BF16 % BF16_PER_VEC == 0, "page bytes must be 16B aligned");
    static_assert(W == NBLOCKS_PER_PAGE * NPAGES_PER_WORKLOAD, "W mismatch");
    // Keep p uniform within a warp so the `p < valid_pages` branch and the
    // winfo_y_indices load are uniform (enables L1 broadcast, no divergence).
    static_assert(PAGE_VEC % WARP_SIZE == 0,
                  "PAGE_VEC must be a multiple of WARP_SIZE for uniform p-per-warp");

    const int tid         = ty * WARP_SIZE + tx;
    const int y_off       = winfo_y_offsets_ptr[workload_id];
    const int y_len       = winfo_y_lens_ptr[workload_id];
    const int valid_pages = (y_len + NBLOCKS_PER_PAGE - 1) / NBLOCKS_PER_PAGE;

    // Flatten (page, vec-in-page) into a single loop so all NUM_THREADS work
    // on every page concurrently instead of 64 threads per page in sequence.
    #pragma unroll
    for (int v = tid; v < TOTAL_VEC; v += NUM_THREADS) {
        const int p   = v / PAGE_VEC;
        const int vip = v - p * PAGE_VEC;
        if (p < valid_pages) {
            const int first_blk =
                winfo_y_indices_ptr[y_off + p * NBLOCKS_PER_PAGE];
            const __nv_bfloat16* src_page =
                y_ptr + static_cast<size_t>(first_blk) * D;
            __nv_bfloat16* dst_page = y_shm_buffer + p * PAGE_BF16;
            cp_async_16(smem_addr(dst_page + vip * BF16_PER_VEC),
                        src_page  + vip * BF16_PER_VEC);
        }
    }
}


// Reduce [G][D] -> [D] (mean over G) into shared memory, parallelized across
// all threads.  Caller must __syncthreads() before consumers read.
template <int G, int D, int W, int NBLOCKS_PER_PAGE, int NPAGES_PER_WORKLOAD>
__device__ __forceinline__ void reduce_x(
const __nv_bfloat16* __restrict__ x_shm_buffer,       // [G][D]
__nv_bfloat16*       __restrict__ x_reduced_shm,      // [D] shared, block-wide
const int            /*phase*/,
const int            /*bx*/,
const int            tx,
const int            ty,
const int            /*tz*/
){
    const int tid     = ty * WARP_SIZE + tx;
    const float inv_G = 1.0f / static_cast<float>(G);

    #pragma unroll
    for (int d = tid; d < D; d += NUM_THREADS) {
        float acc = 0.0f;
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            acc += __bfloat162float(x_shm_buffer[g * D + d]);
        }
        x_reduced_shm[d] = __float2bfloat16(acc * inv_G);
    }
}


// Each warp handles W_PER_WARP = W / NUM_WARPS outputs.  Within a warp the 32
// lanes cooperatively reduce along D via __shfl_xor_sync.  Only lane 0 holds
// the final value for each of its warp's outputs, in o_warp_buffer.
template <int G, int D, int W, int NBLOCKS_PER_PAGE, int NPAGES_PER_WORKLOAD>
__device__ __forceinline__ void compute_gemm(
const __nv_bfloat16* __restrict__ x_reduced_shm,      // [D] shared
const __nv_bfloat16* __restrict__ y_shm_buffer,       // [W][D]
__nv_bfloat16*       __restrict__ o_warp_buffer,      // [W_PER_WARP] per-thread
const int            /*phase*/,
const int            /*bx*/,
const int            tx,
const int            ty,
const int            /*tz*/
){
    constexpr int W_PER_WARP = W / NUM_WARPS;
    constexpr int D_PER_LANE = D / WARP_SIZE;
    static_assert(W % NUM_WARPS  == 0, "W must be a multiple of NUM_WARPS");
    static_assert(D % WARP_SIZE  == 0, "D must be a multiple of WARP_SIZE");

    // Prefetch this lane's slice of x once; reused across W_PER_WARP outputs.
    float x_reg[D_PER_LANE];
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; ++i) {
        x_reg[i] = __bfloat162float(x_reduced_shm[i * WARP_SIZE + tx]);
    }

    #pragma unroll
    for (int wi = 0; wi < W_PER_WARP; ++wi) {
        const int w = ty * W_PER_WARP + wi;
        const __nv_bfloat16* y_row = y_shm_buffer + w * D;

        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            partial += __bfloat162float(y_row[i * WARP_SIZE + tx]) * x_reg[i];
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(0xffffffffu, partial, off);
        }
        if (tx == 0) {
            o_warp_buffer[wi] = __float2bfloat16(partial);
        }
    }
}


template <int G, int D, int W, int NBLOCKS_PER_PAGE, int NPAGES_PER_WORKLOAD>
__device__ __forceinline__ void store_o(
const __nv_bfloat16* __restrict__ o_warp_buffer,      // [W_PER_WARP] per-thread
__nv_bfloat16*       __restrict__ o_ptr,
const int            workload_id,
const int*           __restrict__ winfo_y_offsets_ptr,
const int*           __restrict__ winfo_y_lens_ptr,
const int            /*bx*/,
const int            tx,
const int            ty,
const int            /*tz*/
){
    constexpr int W_PER_WARP = W / NUM_WARPS;
    const int y_off = winfo_y_offsets_ptr[workload_id];
    const int y_len = winfo_y_lens_ptr[workload_id];
    __nv_bfloat16* out_base = o_ptr + y_off;

    if (tx == 0) {
        #pragma unroll
        for (int wi = 0; wi < W_PER_WARP; ++wi) {
            const int w = ty * W_PER_WARP + wi;
            if (w < y_len) {
                out_base[w] = o_warp_buffer[wi];
            }
        }
    }
}


template <int G, int D, int W, int NBLOCKS_PER_PAGE, int NPAGES_PER_WORKLOAD>
__global__ void BlockSparseAttention_Kernel(
const __nv_bfloat16* __restrict__ x_ptr,
const __nv_bfloat16* __restrict__ y_ptr,
const int*           __restrict__ winfo_x_indices_ptr,
const int*           __restrict__ winfo_y_indices_ptr,
const int*           __restrict__ winfo_y_offsets_ptr,
const int*           __restrict__ winfo_y_lens_ptr,
const int*           __restrict__ num_workloads_ptr,
__nv_bfloat16*       __restrict__ o
)
{
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int num_workloads = *num_workloads_ptr;
    const int per_prog      = num_workloads / gridDim.x;
    const int r             = num_workloads % gridDim.x;
    const int start         = bx * per_prog + min(bx, r);
    const int end           = start + per_prog + (bx < r ? 1 : 0);
    const int n_local       = end - start;
    if (n_local <= 0) return;

    constexpr int W_PER_WARP = W / NUM_WARPS;

    __shared__ __align__(16) __nv_bfloat16 x_shm_buffer[2][G][D];
    __shared__ __align__(16) __nv_bfloat16 y_shm_buffer[2][W][D];
    __shared__ __align__(16) __nv_bfloat16 x_reduced_shm[D];
    __align__(16) __nv_bfloat16 o_warp_buffer[W_PER_WARP];

    // Prologue: kick off copies for workload 0 into buffer 0.
    async_copy_x<G, D, W, NBLOCKS_PER_PAGE, NPAGES_PER_WORKLOAD>(
        x_ptr, &x_shm_buffer[0][0][0], start,
        winfo_x_indices_ptr, 0, bx, tx, ty, tz);
    async_copy_y<G, D, W, NBLOCKS_PER_PAGE, NPAGES_PER_WORKLOAD>(
        y_ptr, &y_shm_buffer[0][0][0], start,
        winfo_y_indices_ptr, winfo_y_offsets_ptr, winfo_y_lens_ptr,
        0, bx, tx, ty, tz);
    cp_async_commit();

    #pragma unroll 1
    for (int i = 0; i < n_local; ++i) {
        const int cur      = i & 1;
        const int nxt      = cur ^ 1;
        const int wid      = start + i;
        const int wid_next = wid + 1;

        // Launch next workload's copies before waiting on current, so the
        // compute below overlaps with the next copy group.
        if (i + 1 < n_local) {
            async_copy_x<G, D, W, NBLOCKS_PER_PAGE, NPAGES_PER_WORKLOAD>(
                x_ptr, &x_shm_buffer[nxt][0][0], wid_next,
                winfo_x_indices_ptr, nxt, bx, tx, ty, tz);
            async_copy_y<G, D, W, NBLOCKS_PER_PAGE, NPAGES_PER_WORKLOAD>(
                y_ptr, &y_shm_buffer[nxt][0][0], wid_next,
                winfo_y_indices_ptr, winfo_y_offsets_ptr, winfo_y_lens_ptr,
                nxt, bx, tx, ty, tz);
            cp_async_commit();

            // Keep at most 1 prior group in flight (= the next one); wait
            // until the current group has landed.
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        reduce_x<G, D, W, NBLOCKS_PER_PAGE, NPAGES_PER_WORKLOAD>(
            &x_shm_buffer[cur][0][0], x_reduced_shm, cur, bx, tx, ty, tz);

        // reduce_x writes x_reduced_shm; compute_gemm reads it across warps.
        __syncthreads();

        compute_gemm<G, D, W, NBLOCKS_PER_PAGE, NPAGES_PER_WORKLOAD>(
            x_reduced_shm, &y_shm_buffer[cur][0][0], o_warp_buffer,
            cur, bx, tx, ty, tz);

        store_o<G, D, W, NBLOCKS_PER_PAGE, NPAGES_PER_WORKLOAD>(
            o_warp_buffer, o, wid,
            winfo_y_offsets_ptr, winfo_y_lens_ptr, bx, tx, ty, tz);

        // Make sure all threads are done reading the current shm buffer
        // before the next iteration may consume it (buffer swap).
        __syncthreads();
    }
}
