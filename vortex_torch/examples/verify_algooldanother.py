import os
import sys
from pathlib import Path


def _import_sglang():
    try:
        import sglang as sgl  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import sglang. Install it or add the local checkout's "
            "`sglang/python` directory to PYTHONPATH."
        ) from exc

    if hasattr(sgl, "Engine"):
        return sgl

    # If the repo root contains the SGLang checkout, prefer its Python package
    # over the top-level `sglang/` source directory, which imports as a namespace
    # package and does not expose the public API.
    repo_root = Path(__file__).resolve().parents[2]
    local_sglang_python = repo_root / "sglang" / "python"
    if local_sglang_python.is_dir() and str(local_sglang_python) not in sys.path:
        sys.path.insert(0, str(local_sglang_python))
        sys.modules.pop("sglang", None)
        import sglang as sgl  # type: ignore
        if hasattr(sgl, "Engine"):
            return sgl

    raise RuntimeError(
        "Imported `sglang`, but it does not expose `Engine`. "
        "This usually means PYTHONPATH points at the repository root `sglang/` "
        "directory instead of `sglang/python`. "
        f"Tried local path: {local_sglang_python}"
    )


sgl = _import_sglang()
import vortex_torch
from transformers import AutoTokenizer, AutoConfig
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    MultilingualExtractiveMatchMetric
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.models.model_output import ModelResponse
from datasets import load_dataset, Dataset, concatenate_datasets
import argparse
import json
from datetime import datetime 
from vortex_torch.indexer.utils_sglang import SCHEDULE_POLICY_ALIASES

def verify_algos(
trials: int = 2,
topk_val: int = 29,
topk_ratio: float = 0.0625,
block_size: int = 16,
page_size: int = 256,
workload_chunk_size: int = 32,
max_input_length: int = 4096,
generation_max_new_tokens: int = 8192,
vortex_module_name: str = "gqa_block_sparse_attention",
model_name: str = "Qwen/Qwen3-1.7B",
sparse_attention: bool = True,
mem: float = 0.8,
data_path: str = "examples/amc23.jsonl",
tp_size: int = 1,
policy_name: str = None, 
): 
    schedule = {
        "qwen3-1.7b-0.86": {0: 29, 2000: 61, 4000: 93, 6000: 93, 8000: 125, 10000: 157, 12000: 189, 14000: 221, 16000: 221, 18000: 253, 20000: 253, 22000: 253, 24000: 253, 26000: 253, 28000: 253, 30000: 253, 32000: 253, 34000: 253, 36000: 253, 38000: 253, 40000: 253}, 
        "qwen3-1.7b-0.88": {0: 45, 2000: 77, 4000: 125, 6000: 157, 8000: 157, 10000: 189, 12000: 221, 14000: 253, 16000: 253, 18000: 317, 20000: 317, 22000: 317, 24000: 317, 26000: 317, 28000: 317, 30000: 317, 32000: 317, 34000: 317, 36000: 317, 38000: 317, 40000: 317}, 
        "qwen3-1.7b-0.90": {0: 45, 2000: 93, 4000: 157, 6000: 189, 8000: 221, 10000: 253, 12000: 317, 14000: 317, 16000: 381, 18000: 381, 20000: 381, 22000: 381, 24000: 381, 26000: 381, 28000: 381, 30000: 381, 32000: 381, 34000: 381, 36000: 381, 38000: 381, 40000: 381}, 
        "qwen3-1.7b-0.92": {0: 61, 2000: 125, 4000: 189, 6000: 221, 8000: 253, 10000: 317, 12000: 381, 14000: 381, 16000: 445, 18000: 509, 20000: 509, 22000: 509, 24000: 509, 26000: 509, 28000: 509, 30000: 509, 32000: 509, 34000: 509, 36000: 509, 38000: 509, 40000: 509}, 
        "qwen3-1.7b-0.94": {0: 77, 2000: 157, 4000: 221, 6000: 317, 8000: 381, 10000: 445, 12000: 509, 14000: 509, 16000: 637, 18000: 765, 20000: 765, 22000: 765, 24000: 765, 26000: 765, 28000: 765, 30000: 765, 32000: 765, 34000: 765, 36000: 765, 38000: 765, 40000: 765}, 
        # "debugging-0.9": {0: debuggingvalue, 2000: debuggingvalue, 4000: debuggingvalue, 6000: debuggingvalue, 8000: debuggingvalue, 10000: debuggingvalue, 12000: debuggingvalue, 14000: debuggingvalue, 16000: debuggingvalue, 18000: debuggingvalue, 20000: debuggingvalue, 22000: debuggingvalue, 24000: debuggingvalue, 26000: debuggingvalue, 28000: debuggingvalue, 30000: debuggingvalue, 32000: debuggingvalue, 34000: debuggingvalue, 36000: debuggingvalue, 38000: debuggingvalue, 40000: debuggingvalue}, 
    } 
    
    sparsity_policies = dict(SCHEDULE_POLICY_ALIASES)

    llm = sgl.Engine(model_path=model_name, 
                    disable_cuda_graph=False,
                    vortex_block_size=block_size,
                    page_size=page_size,
                    vortex_topk_val=topk_val,
                    tp_size=tp_size,
                    disable_overlap_schedule=True,
                    attention_backend="flashinfer",
                    enable_vortex_sparsity=sparse_attention,
                    vortex_block_reserved_bos=1,
                    vortex_block_reserved_eos=2,
                    vortex_topk_ratio=topk_ratio,
                    vortex_layers_skip=list(range(1)),
                    vortex_module_name=vortex_module_name,
                    vortex_max_seq_lens=max_input_length + generation_max_new_tokens,
                    mem_fraction_static=mem,
                    vortex_workload_chunk_size=max(page_size // block_size, workload_chunk_size),
                    vortex_compilation_cache_dir="~/.vortex_compilation_cache",
                    context_length=40960,
                    vortex_schedule_policy=policy_name, 
    ) 
    
    with open(data_path, "r", encoding="utf-8") as f:
        requests = [json.loads(line) for line in f]
    
    requests = requests * trials
    prompts = [req["prompt"] for req in requests]

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": generation_max_new_tokens}
    
    o = llm.generate(prompts, sampling_params)
    gold_metric =  MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(ExprExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
            aggregation_function=max,
        )
    
    results = []
    for data, item in zip(requests, o):
        golds = [data["answer"]]
        target = Doc(query=data["question"],choices=golds, gold_index=0)
        predictions = item["text"]
        try:
            result = gold_metric.compute(model_response=ModelResponse(text=[predictions]), doc=target)
        except:
            result = 0.0
        
        results.append(
            {
                "score": float(result),
                "prediction": [predictions],
                "choices": golds,
                "query": data["question"],
                "e2e_latency": item["meta_info"]["e2e_latency"],
                "num_tokens": item["meta_info"]["completion_tokens"]
            }
        )
    

    total_accuracy = 0.0
    total_tokens = 0
    e2e_time = 0
    count = 0
    unique_result = {}

    for item in results:
        total_accuracy += item['score']
        count += 1
        total_tokens += item["num_tokens"]
        e2e_time = max(e2e_time, item["e2e_latency"])
        if item['query'] not in unique_result:
            unique_result[item['query']] = item["score"]
        else:
            unique_result[item['query']] = max(item["score"], unique_result[item['query']])


    global_summary = {
        f'mean@{trials}': total_accuracy / count if count > 0 else 0,
        f'pass@{trials}': sum(unique_result.values()) / len(unique_result),
        'total_example': count,
        "e2e_time": e2e_time,
        "total_tokens": total_tokens, 
        "throughput": total_tokens / e2e_time,
    }
    
    return global_summary

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vortex_torch verify_algos benchmark."
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=2,
        help="Number of trials to run (default: 2).",
    )

    parser.add_argument(
        "--topk-val",
        type=int,
        default=29,
        help="Top-k value to use in the algorithm (default: 30).",
    )
    
    parser.add_argument(
        "--topk-ratio",
        type=float,
        default=0.0625,
        help="Top-k ratio to use in the algorithm (default: 0.0625).",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block Size for Sglang (default: 16).",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=256,
        help="Page Size for Sglang (default: 256).",
    )

    parser.add_argument(
        "--workload-chunk-size",
        type=int,
        default=32,
        help="Workload Chunk Size for Sglang (default: 32).",
    )

    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=8192,
        help="Max new tokens to generate (default: 8192).",
    )

    parser.add_argument(
        "--max-input-length",
        type=int,
        default=4096,
        help="Max input tokens (default: 4096).",
    )

    parser.add_argument(
        "--vortex-module-name",
        type=str,
        default="gqa_block_sparse_attention",
        help='Name of the vortex module to test (default: "gqa_block_sparse_attention").',
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help='HuggingFace model name to load (default: "Qwen/Qwen3-1.7B").',
    )

    parser.add_argument(
        "-f", "--full-attention",
        action="store_true",
        help="Use full attention instead of vortex sparse attention.",
    )

    parser.add_argument(
        "--mem",
        type=float,
        default=0.8,
        help="memory fraction in sglang",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="examples/amc23.jsonl",
        help="Path to the evaluation data (default: examples/amc23.jsonl).",
    )

    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for Sglang (default: 1).",
    ) 
    
    parser.add_argument(
        "--policy-name",
        type=str,
        default=None,
        help="Policy name to use in the algorithm (default: None).",
    ) 

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args() 
    print(args) 
    print("--------------------------------") 
    print("\n") 
    print(f"Policy: {args.policy_name}") 
    print("\n") 
    print("--------------------------------") 
    print("\n") 

    summary = verify_algos(
        trials=args.trials,
        topk_val=args.topk_val,
        topk_ratio=args.topk_ratio,
        block_size=args.block_size,
        page_size=args.page_size,
        workload_chunk_size=args.workload_chunk_size,
        generation_max_new_tokens=args.generation_max_new_tokens,
        max_input_length=args.max_input_length,
        vortex_module_name=args.vortex_module_name,
        model_name=args.model_name,
        sparse_attention=(args.vortex_module_name != "full_attention"),
        mem=args.mem,
        # data_path = "examples/amc23.jsonl", 
        data_path = args.data_path, 
        tp_size=args.tp_size,
        policy_name=args.policy_name,
    ) 
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"summary_ratio/{args.model_name.replace('/', '_')}_{args.vortex_module_name}_{args.trials}trials_tp{args.tp_size}_{current_time}.json"
    os.makedirs("summary_ratio", exist_ok=True)
    ## args to summary
    summary["args"] = {
        "trials": args.trials,
        "topk_val": args.topk_val,
        "topk_ratio": args.topk_ratio,
        "block_size": args.block_size,
        "page_size": args.page_size,
        "workload_chunk_size": args.workload_chunk_size,
        "generation_max_new_tokens": args.generation_max_new_tokens,
        "max_input_length": args.max_input_length,
        "vortex_module_name": args.vortex_module_name,
        "model_name": args.model_name,
        "sparse_attention": (args.vortex_module_name != "full_attention"),
        "mem": args.mem,
        "data_path": args.data_path,
        "tp_size": args.tp_size,
    }
    with open(output_path, "w", encoding="utf-8") as f: 
        print(summary) 
        json.dump(summary, f, ensure_ascii=False, indent=4) 
    
