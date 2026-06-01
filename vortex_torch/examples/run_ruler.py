import json
import sys
import sglang as sgl
from transformers import AutoTokenizer
def main():
    model_name = "Qwen/Qwen3-4B"

    default_policy = r"""
const int static_kv_budget = topk_val + block_reserved_bos + block_reserved_eos;
const int dynamic_kv_budget = int(cached_block_len * topk_ratio);
return max(static_kv_budget, dynamic_kv_budget);
"""

    llm = sgl.Engine(model_path=model_name, 
                    disable_cuda_graph=False,
                    page_size=16,
                    vortex_block_size=16,
                    vortex_topk_val=29,
                    disable_overlap_schedule=True,
                    kv_cache_dtype="fp8_e4m3",
                    attention_backend="flashinfer",
                    vortex_schedule_policy=default_policy,
                    enable_vortex_sparsity=True,
                    vortex_block_reserved_bos=1,
                    vortex_block_reserved_eos=2,
                    vortex_layers_skip=list(range(1)),
                    vortex_module_name="gqa_quest_sparse_attention",
                    vortex_max_seq_lens=8192,
                    mem_fraction_static=0.8,
                    vortex_workload_chunk_size=32,
                    vortex_compilation_cache_dir="~/.vortex_compilation_cache",
                    )
    
    with open("examples/validation.jsonl", "r", encoding="utf-8") as f:
        ruler_data = [json.loads(line)["input"] for line in f]

    with open("examples/validation.jsonl", "r", encoding="utf-8") as f:
        ruler_outputs = [json.loads(line)["outputs"][0] for line in f]
    
    texts = [
        [{"role":"user","content": x}] for x in ruler_data
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [
        tokenizer.apply_chat_template(
        text,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for text in texts
    ]
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 64}
    accuracy = 0
    with open("examples/ruler_output.jsonl", "w", encoding="utf-8") as f:
            o = llm.generate(prompts, sampling_params)
            for res, answer in zip(o, ruler_outputs):
                    json.dump(res, f, ensure_ascii=False)
                    f.write("\n")
                    if answer in res["text"]:
                        accuracy += 1.0
    print(f"Ruler Accuracy: {accuracy / len(ruler_outputs) * 100:.2f}%")

if __name__ == "__main__":
    main()
