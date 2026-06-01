#!/usr/bin/env python3
import argparse
import dataclasses
import logging
from pathlib import Path
from typing import List, Optional

import torch 
import time 

from sglang.bench_one_batch import decode, extend, load_model
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger 

'''
python profile_decode.py \
  --model-path Qwen/Qwen3-1.7B \
  --batch-size 2 \
  --max-new-tokens 64 \
  --trace-path decode_trace.json \
  --summary-path decode_summary.txt \
  --record-shapes \
  --profile-memory \
  --with-stack \
  --vortex-algorithm BLOCK_TOPK \
  --vortex-num-selected-pages 29 
''' 


def _read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt_file:
        return Path(prompt_file).read_text()
    if prompt:
        return prompt
    return """<FILL YOUR PROMPT HERE>"""


def _parse_int_list(csv: Optional[str]) -> Optional[List[int]]:
    if not csv:
        return None
    return [int(x) for x in csv.split(",") if x.strip()]


def _build_reqs(prompt: str, batch_size: int, tokenizer, max_new_tokens: int): 
    prompts = [prompt] * batch_size
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(temperature=0, max_new_tokens=max_new_tokens)

    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req) 
        
        # TODO: print the length of the requestin tokens 
        print(f"Request length (tokens): {len(input_ids[0]) if input_ids else 0}") 
    return reqs


def _prefill_with_optional_chunking(reqs, model_runner, prefill_chunk_size: int):
    """Run prefill as one shot or in token chunks per request."""
    if prefill_chunk_size <= 0:
        return extend(reqs, model_runner)

    input_lens = [len(req.origin_input_ids) for req in reqs]
    if len(set(input_lens)) != 1:
        raise ValueError(
            "Chunked prefill in this script requires equal prompt lengths across the batch."
        )

    total_len = input_lens[0]
    if prefill_chunk_size >= total_len:
        return extend(reqs, model_runner)

    # Keep per-request prefix KV indices across chunks.
    prefix_indices = [[] for _ in reqs]
    next_token_ids = None
    next_token_logits = None
    final_batch = None

    chunk_start = 0
    while chunk_start < total_len:
        chunk_end = min(chunk_start + prefill_chunk_size, total_len)

        for i, req in enumerate(reqs):
            req.prefix_indices = prefix_indices[i]
            req.fill_ids = req.origin_input_ids[:chunk_end]
            req.extend_input_len = chunk_end - chunk_start
            req.logprob_start_len = len(req.origin_input_ids) - 1

        try:
            next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
        except AttributeError as e:
            # SGLang may raise AttributeError in OOM path when tree_cache is None.
            if "evictable_size" in str(e):
                available_tokens = model_runner.token_to_kv_pool_allocator.available_size()
                raise RuntimeError(
                    "Prefill KV allocation failed (OOM). "
                    f"available_kv_tokens={available_tokens}. "
                    "This error was originally masked by a tree_cache=None bug in SGLang."
                ) from e
            raise

        is_last_chunk = chunk_end == total_len
        req_pool_indices = batch.req_pool_indices.detach().cpu().tolist()

        if is_last_chunk:
            final_batch = batch
        else:
            # Persist prefix KV indices and free request slots for the next chunk.
            for i, req_pool_idx in enumerate(req_pool_indices):
                prefix_indices[i] = model_runner.req_to_token_pool.req_to_token[
                    req_pool_idx, :chunk_end
                ].clone()
            model_runner.req_to_token_pool.free(req_pool_indices)

        chunk_start = chunk_end

    return next_token_ids, next_token_logits, final_batch


def _validate_kv_budget(reqs, model_runner, max_new_tokens: int):
    batch_size = len(reqs)
    prompt_tokens = sum(len(r.origin_input_ids) for r in reqs)
    decode_tokens = max(max_new_tokens - 1, 0) * batch_size
    required_tokens = prompt_tokens + decode_tokens
    capacity = model_runner.max_total_num_tokens
    if required_tokens > capacity:
        raise ValueError(
            "Requested workload exceeds KV capacity: "
            f"required_tokens={required_tokens} (prompt={prompt_tokens}, decode={decode_tokens}) "
            f"> max_total_num_tokens={capacity}. "
            "Chunked prefill reduces activation peak, but not total KV tokens. "
            "Lower batch size / prompt length / max-new-tokens, or increase KV budget."
        )


def _print_kv_pool_stats(model_runner, stage: str):
    allocator = model_runner.token_to_kv_pool_allocator
    capacity = model_runner.max_total_num_tokens
    available = allocator.available_size()
    used = capacity - available

    print(
        f"[{stage}] KV pool tokens: used={used}, available={available}, capacity={capacity}"
    )

    # Hybrid allocators expose split pools with separate capacities.
    if hasattr(allocator, "size_full") and hasattr(allocator, "size_swa"):
        full_avail = allocator.full_available_size()
        swa_avail = allocator.swa_available_size()
        print(
            f"[{stage}] KV pool split: "
            f"full_used={allocator.size_full - full_avail}, "
            f"full_capacity={allocator.size_full}, "
            f"swa_used={allocator.size_swa - swa_avail}, "
            f"swa_capacity={allocator.size_swa}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile decode-only kernels with SGLang VTX FlashInfer backend"
    )
    ServerArgs.add_cli_args(parser)

    parser.add_argument("--prompt", default=None, help="Prompt string")
    parser.add_argument("--prompt-file", default=None, help="Path to prompt file")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)

    parser.add_argument("--trace-path", default="profile_decode_trace.json")
    parser.add_argument("--summary-path", default="profile_decode_summary.txt")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--profile-memory", action="store_true") 
    parser.add_argument("--sequence-length", type=str, choices=["mean", "8000", "16000", "24000", "32000"], default="32000") 
    parser.add_argument("--enableit", type=int, default=0) 
    parser.add_argument("--page-siz3", type = int, default = 16) 

    parser.add_argument("--vortex-algorithm", default="BLOCK_TOPK") 
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=-1,
        help="If >0, prefill each request in token chunks of this size to reduce peak memory.",
    ) 
    parser.add_argument("--decodingsteps", type=int, default=100) 

    args = parser.parse_args()
    # Backfill any ServerArgs fields that are not exposed by add_cli_args in this build.
    for field in dataclasses.fields(ServerArgs):
        if hasattr(args, field.name):
            continue
        if field.default is not dataclasses.MISSING:
            setattr(args, field.name, field.default)
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
            setattr(args, field.name, field.default_factory())  # type: ignore[misc]
        else:
            setattr(args, field.name, None) 
    resolved_vortex_topk_val = getattr(args, "vortex_topk_val", 5)
    print(
        "--------------------------------\n"
        f"enableit: {args.enableit} "
        f"vortex_topk_val: {resolved_vortex_topk_val} "
        f"pagesize: {args.page_siz3} "
        f"batchsize: {args.batch_size} "
        f"decodingsteps: {args.decodingsteps}\n"
        "--------------------------------"
    ) 
    server_args = ServerArgs.from_cli_args(args)

    # Force VTX FlashInfer backend
    server_args.attention_backend = "flashinfer"
    if args.enableit == 1: 
        server_args.enable_vortex_sparsity = True 
    else: 
        server_args.enable_vortex_sparsity = False 
    # server_args.enable_vortex_sparsity = False 
    server_args.vortex_module_name = "block_sparse_attention" 
    server_args.vortex_topk_val = resolved_vortex_topk_val 
    server_args.vortex_layers_skip = [0, 1] 
    server_args.page_size = args.page_siz3 
    server_args.vortex_page_reserved_bos = 1 
    server_args.vortex_page_reserved_eos = 2 
    # server_args.vortex_max_seq_lens = args.vortex_max_seq_lens 
    server_args.vortex_max_seq_lens = 40000 
    server_args.disable_cuda_graph = False 
    server_args.disable_overlap_schedule = True 

    # if getattr(args, "disable_cuda_graph", False): 
        # server_args.disable_cuda_graph = True 

    if server_args.tp_size != 1:
        raise ValueError("This script supports tp_size=1 only. Use tp_size=1 for profiling.")

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    _set_envs_and_config(server_args)
    configure_logger(server_args, prefix=" TP0")

    port_args = PortArgs.init_new(server_args)
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank=0)

    # Print the resolved attention backend to avoid ambiguity during profiling.
    backend_obj = getattr(model_runner, "attn_backend", None)
    backend_cls = backend_obj.__class__ if backend_obj is not None else None
    print(
        "Resolved attention backend:",
        {
            "attention_backend_arg": server_args.attention_backend,
            "enable_vortex_sparsity": server_args.enable_vortex_sparsity,
            "class": f"{backend_cls.__module__}.{backend_cls.__name__}"
            if backend_cls is not None
            else "<none>",
        },
    )

    # Clear pools for a clean run
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    if args.sequence_length == "mean": 
        prompt = "Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering. The following is the problem. A firecracker was thrown vertically upward with a speed of $20 \, \mathrm{m/s}$. One second after the flight began, it exploded into two unequal parts, with their mass ratio being $1:2$. Immediately after the explosion, the smaller fragment flew horizontally with a speed of $16 \, \mathrm{m/s}$. Find the speed (in m/s) of the second fragment immediately after the explosion. Assume the acceleration due to gravity is $10 \, \mathrm{m/s}^2$. Begin your solution." 
    else: 
        if args.sequence_length == "8000": 
            filename = "sixk.txt" 
        elif args.sequence_length == "16000": 
            filename = "sixteenk.txt" 
        elif args.sequence_length == "24000": 
            filename = "twentyfourk.txt" 
        elif args.sequence_length == "32000": 
            filename = "thirtyk.txt" 
        prompt = _read_prompt(prompt = None, prompt_file = filename) 
    reqs = _build_reqs(prompt, args.batch_size, tokenizer, args.max_new_tokens) 
    _validate_kv_budget(reqs, model_runner, args.max_new_tokens) 

    with torch.no_grad():
        # Prefill (extend) without profiling
        next_token_ids, _, batch = _prefill_with_optional_chunking(
            reqs, model_runner, args.prefill_chunk_size
        )
        _print_kv_pool_stats(model_runner, stage="after_prefill")

        if server_args.device == "cuda":
            torch.cuda.synchronize() 

        decode_steps = args.decodingsteps + 10 

        if server_args.device == "cuda":
            torch.cuda.synchronize() 
        
        # warmup 
        for _ in range(10):
            next_token_ids, _ = decode(next_token_ids, batch, model_runner) 

        tic = time.perf_counter() 
        for step in range(decode_steps - 10): 
            next_token_ids, _ = decode(next_token_ids, batch, model_runner) 
        torch.cuda.synchronize() 
        toc = time.perf_counter() 
        print(f"Time taken: {toc - tic} seconds\nAverage time per step: {(toc - tic) / ((decode_steps - 10) * 1000)} ms") 


if __name__ == "__main__":
    main() 
