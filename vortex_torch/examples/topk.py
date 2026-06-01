import torch
import triton
from vortex_torch_C import topk_output_v2, topk_output
from tqdm import tqdm
SEQ_LENS = [1024, 1536, 2048, 4096]
BATCH_SIZES = [16, 32, 64, 128, 256, 512]

K = 61
EVAL_KS = [5, 10, 20, 32, 48, 61]

RESERVE_BOS = 0
RESERVE_EOS = 0
DEVICE = "cuda"
NUM_RUNS = 10


def make_inputs(batch_size, seq_len, k, reserve_bos, reserve_eos, device="cuda"):
    dense_kv_indptr = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32, device=device
    )

    dense_kv_indices = torch.arange(
        0, batch_size * seq_len, dtype=torch.int32, device=device
    )

    scores = torch.randn(
        batch_size * seq_len, dtype=torch.bfloat16, device=device
    )

    sparse_kv_indptr = torch.arange(
        0, batch_size * k + 1, k, dtype=torch.int32, device=device
    )

    sparse_kv_indices_v1 = torch.empty(
        batch_size * k, dtype=torch.int32, device=device
    )

    sparse_kv_indices_v2 = torch.empty(
        batch_size * k, dtype=torch.int32, device=device
    )

    return (
        scores,
        dense_kv_indptr,
        dense_kv_indices,
        sparse_kv_indptr,
        sparse_kv_indices_v1,
        sparse_kv_indices_v2,
    )


def run_v1(
    scores,
    dense_kv_indptr,
    dense_kv_indices,
    sparse_kv_indptr,
    sparse_kv_indices,
    batch_size,
    k,
    reserve_bos,
    reserve_eos,
    seq_len,
):
    topk_output(
        scores,
        dense_kv_indptr,
        dense_kv_indices,
        sparse_kv_indptr,
        sparse_kv_indices,
        batch_size,
        reserve_bos,
        reserve_eos,
        seq_len,
    )


def run_v2(
    scores,
    dense_kv_indptr,
    dense_kv_indices,
    sparse_kv_indptr,
    sparse_kv_indices,
    batch_size,
    k,
    reserve_bos,
    reserve_eos,
    seq_len,
):
    topk_output_v2(
        scores,
        dense_kv_indptr,
        sparse_kv_indptr,
        dense_kv_indices,
        sparse_kv_indices,
        batch_size,
        reserve_bos,
        reserve_eos,
        seq_len,
    )


def compute_recall_at_ks(ref_indices, pred_indices, batch_size, k, eval_ks):
    """
    ref_indices: v1 output, ordered top-k, shape [batch_size * k]
    pred_indices: v2 output, unordered k candidates, shape [batch_size * k]

    recall@r = fraction of ref top-r recovered in pred top-k
    """
    ref = ref_indices.view(batch_size, k).to(torch.int64)   # [B, K]
    pred = pred_indices.view(batch_size, k).to(torch.int64) # [B, K]

    recalls = {}

    for r in eval_ks:
        ref_r = ref[:, :r]  # only reference top-r
        # compare ref top-r against ALL pred K entries
        match = (ref_r.unsqueeze(2) == pred.unsqueeze(1))   # [B, r, K]
        hit = match.any(dim=2)                              # [B, r]
        recall = hit.float().sum(dim=1) / r                 # [B]
        recalls[r] = recall.mean().item()

    return recalls


def bench_latency(fn):
    # warmup
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(
        fn,
        warmup=100,
        rep=1000,
        return_mode="mean",
    )
    return ms


def eval_one(batch_size, seq_len, k, reserve_bos, reserve_eos):
    (
        scores,
        dense_kv_indptr,
        dense_kv_indices,
        sparse_kv_indptr,
        sparse_kv_indices_v1,
        sparse_kv_indices_v2,
    ) = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        k=k,
        reserve_bos=reserve_bos,
        reserve_eos=reserve_eos,
        device=DEVICE,
    )

    def fn_v1():
        run_v1(
            scores,
            dense_kv_indptr,
            sparse_kv_indptr,
            dense_kv_indices,
            sparse_kv_indices_v1,
            batch_size,
            k,
            reserve_bos,
            reserve_eos,
            seq_len,
        )

    def fn_v2():
        run_v2(
            scores,
            dense_kv_indptr,
            dense_kv_indices,
            sparse_kv_indptr,
            sparse_kv_indices_v2,
            batch_size,
            k,
            reserve_bos,
            reserve_eos,
            seq_len,
        )

    # 先各跑一次，拿输出
    fn_v1()
    fn_v2()
    torch.cuda.synchronize()
    recalls = compute_recall_at_ks(
        sparse_kv_indices_v1,
        sparse_kv_indices_v2,
        batch_size=batch_size,
        k=k,
        eval_ks=EVAL_KS,
    )

    # latency
    ms_v1 = bench_latency(fn_v1)
    ms_v2 = bench_latency(fn_v2)

    return ms_v1, ms_v2, recalls


def main():
    torch.cuda.init()

    results = {}

    for bs in BATCH_SIZES:
        results[bs] = {}
        for seq_len in SEQ_LENS:
            agg = {
                "v1_ms": [],
                "v2_ms": [],
                5: [],
                10: [],
                20: [],
                32: [],
                48: [],
                61: [],
            }

            for seed in tqdm(range(NUM_RUNS)):
                torch.cuda.manual_seed(seed)
                ms_v1, ms_v2, recalls = eval_one(
                    batch_size=bs,
                    seq_len=seq_len,
                    k=K,
                    reserve_bos=RESERVE_BOS,
                    reserve_eos=RESERVE_EOS,
                )
                agg["v1_ms"].append(ms_v1)
                agg["v2_ms"].append(ms_v2)
                for kk in EVAL_KS:
                    agg[kk].append(recalls[kk])

            result = {
                "v1_ms_mean": sum(agg["v1_ms"]) / len(agg["v1_ms"]),
                "v2_ms_mean": sum(agg["v2_ms"]) / len(agg["v2_ms"]),
                "speedup": (sum(agg["v1_ms"]) / len(agg["v1_ms"])) / (sum(agg["v2_ms"]) / len(agg["v2_ms"])),
                "recall@5": sum(agg[5]) / len(agg[5]),
                "recall@10": sum(agg[10]) / len(agg[10]),
                "recall@20": sum(agg[20]) / len(agg[20]),
                "recall@32": sum(agg[32]) / len(agg[32]),
                "recall@48": sum(agg[48]) / len(agg[48]),
                "recall@61": sum(agg[61]) / len(agg[61]),
            }
            results[bs][seq_len] = result

            print(
                f"bs={bs:>4}, seq_len={seq_len:>6} | "
                f"v1={result['v1_ms_mean']:.6f} ms, "
                f"v2={result['v2_ms_mean']:.6f} ms, "
                f"speedup={result['speedup']:.4f}x | "
                f"R@5={result['recall@5']:.6f}, "
                f"R@10={result['recall@10']:.6f}, "
                f"R@20={result['recall@20']:.6f}, "
                f"R@32={result['recall@32']:.6f}, "
                f"R@48={result['recall@48']:.6f}, "
                f"R@61={result['recall@61']:.6f}"
            )

    print("\nSummary:")
    for bs in BATCH_SIZES:
        for seq_len in SEQ_LENS:
            r = results[bs][seq_len]
            print(
                f"[bs={bs}, seq={seq_len}] "
                f"v1={r['v1_ms_mean']:.6f} ms, "
                f"v2={r['v2_ms_mean']:.6f} ms, "
                f"speedup={r['speedup']:.4f}x, "
                f"R@5={r['recall@5']:.6f}, "
                f"R@10={r['recall@10']:.6f}, "
                f"R@20={r['recall@20']:.6f}, "
                f"R@32={r['recall@32']:.6f}, "
                f"R@48={r['recall@48']:.6f}, "
                f"R@61={r['recall@61']:.6f}"
            )


if __name__ == "__main__":
    main()