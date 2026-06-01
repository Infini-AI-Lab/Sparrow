#!/usr/bin/env python3
import argparse
import dataclasses
import logging
from pathlib import Path
from typing import List, Optional

import random

import torch

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


def _build_reqs(
    prompt: str,
    batch_size: int,
    tokenizer,
    max_new_tokens: int,
    input_len: Optional[int] = None,
    random_seed: int = 0,
):
    sampling_params = SamplingParams(temperature=0, max_new_tokens=max_new_tokens)

    if input_len is not None:
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        rng = random.Random(random_seed)
        prompts = [""] * batch_size
        input_ids = [
            [rng.randrange(vocab_size) for _ in range(input_len)]
            for _ in range(batch_size)
        ]
    else:
        prompts = [prompt] * batch_size
        input_ids = [tokenizer.encode(p) for p in prompts]

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

        print(f"Request {i} length (tokens): {len(input_ids[i])}")
    return reqs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile decode-only kernels with SGLang VTX FlashInfer backend"
    )
    ServerArgs.add_cli_args(parser)

    parser.add_argument("--prompt", default=None, help="Prompt string")
    parser.add_argument("--prompt-file", default=None, help="Path to prompt file")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="If set, ignore prompt and use randomly-initialized token ids of this length.",
    )
    parser.add_argument("--trace-path", default="profile_decode_trace.json")
    parser.add_argument("--summary-path", default="profile_decode_summary.txt")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")

    parser.add_argument("--vortex-algorithm", default="BLOCK_TOPK") 

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
    server_args = ServerArgs.from_cli_args(args)

    # Force VTX FlashInfer backend
    server_args.attention_backend = "flashinfer"
    server_args.enable_vortex_sparsity = False
    server_args.disable_overlap_schedule = True
    server_args.disable_cuda_graph = False
    server_args.vortex_module_name = "block_sparse_attention" 
    server_args.vortex_topk_val = 29
    server_args.vortex_layers_skip = [0] 
    server_args.page_size = 512
    server_args.block_size = 16
    server_args.vortex_block_reserved_bos = 1
    server_args.vortex_block_reserved_eos = 2
    server_args.vortex_workload_chunk_size = 32
    server_args.vortex_compilation_cache_dir="./vortex_compilation_cache"
    server_args.vortex_max_seq_lens = 20480
    server_args.mem_fraction_static = 0.85
    server_args.model_path = "Qwen/Qwen3-1.7B"
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

    if args.input_len is not None:
        prompt = ""
    else:
        prompt = _read_prompt(prompt=args.prompt, prompt_file=args.prompt_file or "long_prompt.txt")
    reqs = _build_reqs(
        prompt,
        args.batch_size,
        tokenizer,
        args.max_new_tokens,
        input_len=args.input_len,
        random_seed=args.random_seed,
    )

    with torch.no_grad():
        # Prefill (extend) without profiling
        next_token_ids, _, batch = extend(reqs, model_runner)

        if server_args.device == "cuda":
            torch.cuda.synchronize() 

        decode_steps = max(args.max_new_tokens - 1, 0)

        if server_args.device == "cuda":
            torch.cuda.synchronize()

        # Optional: tell nsys to only start collecting here
        if server_args.device == "cuda":
            torch.cuda.cudart().cudaProfilerStart()

        with torch.cuda.nvtx.range("decode_loop"):
            for step in range(decode_steps):
                with torch.cuda.nvtx.range(f"decode_step_{step}"):
                    next_token_ids, _ = decode(next_token_ids, batch, model_runner)

        if server_args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop() 


if __name__ == "__main__":
    main() 
