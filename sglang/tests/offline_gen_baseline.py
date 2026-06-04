import json
import sys
sys.path.append("../")
import python.sglang as sgl
from transformers import AutoTokenizer
import time
def main():
    model_name = "Qwen/Qwen3-1.7B"
    llm = sgl.Engine(model_path=model_name, 
                    disable_cuda_graph=False,
                    page_size=16,
                    vortex_topk_val=30,   
                    disable_overlap_schedule=True,
                    attention_backend="flashinfer",
                    enable_vortex_sparsity=False,
                    vortex_page_reserved_bos=1,
                    vortex_page_reserved_eos=1,
                    vortex_layers_skip=list(range(1)),
                    vortex_module_name="block_sparse_attention",
                    vortex_max_seq_lens=8192,
                    mem_fraction_static=0.9
                    )
    
    with open('story.txt', "r", encoding="utf-8") as file:
        content = file.read()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = [
        [{"role":"user","content": content}],
    ]
    
    prompts = [
        tokenizer.apply_chat_template(
        text,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    ) for text in texts
    ]
    
    prompts = prompts * 8
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 2048}

    total_tokens = 0
    total_time = 0.0
    e2e_time = 0

    start = time.perf_counter()

    o = llm.generate(prompts, sampling_params)

    elapsed = time.perf_counter() - start
    total_time += elapsed

    with open("output.jsonl", "w", encoding="utf-8") as f:
        for item in o:
            total_tokens += item["meta_info"]["completion_tokens"] 
            e2e_time = max(e2e_time, item["meta_info"]["e2e_latency"])
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

        meta_data = {"e2e_time": e2e_time, "total_time": total_time, "total_tokens": total_tokens, "throughput": total_tokens / total_time}
        print(meta_data)

if __name__ == "__main__":
    main()
