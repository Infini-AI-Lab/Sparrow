from transformers import Qwen3ForCausalLM
import json
from transformers import AutoTokenizer
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from BSA.nsa import nsa_func
from typing import Optional

model_name = "Qwen/Qwen3-1.7B"

def nsa_attention(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
):      
        num_key_value_heads = key_states.shape[-3]
        num_attention_heads = query_states.shape[-3]
        head_dim = query_states.shape[-1]
        cu_seqlens = torch.tensor([i * query_states.shape[2] for i in range(query_states.shape[0] + 1)], device=query_states.device, dtype=torch.int32)
        
        query_states = query_states.transpose(1, 2).contiguous()
        query_states = query_states.reshape(1, -1, num_attention_heads, head_dim)
        
        key_states = key_states.transpose(1, 2).contiguous()
        key_states = key_states.reshape(1, -1, num_key_value_heads, head_dim)
        value_states = value_states.transpose(1, 2).contiguous()
        value_states = value_states.reshape(1, -1, num_key_value_heads, head_dim)
        
    
        
        attn_output  = nsa_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens=cu_seqlens,
            block_size=16,
            block_counts=32
        )
        
        return attn_output, None
    
ALL_ATTENTION_FUNCTIONS["nsa"] = nsa_attention

with open("/scratch/zhuoming/validation.jsonl", "r", encoding="utf-8") as f:
    ruler_data = [json.loads(line)["input"] for line in f]
    
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

tokenizer.padding_side = "right"

batch = tokenizer(
    prompts,                   
    return_tensors="pt",       
    padding=True,            
    truncation=False,       
    add_special_tokens=False
)


qwen3 = Qwen3ForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16").cuda()
qwen3_nsa = Qwen3ForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", _attn_implementation="nsa").cuda()

with torch.inference_mode():
    for i in range(50):
        batch = tokenizer(
            prompts[2 * i: 2 * i+ 2],                  
            return_tensors="pt",       
            padding=True,            
            truncation=False,       
            add_special_tokens=False
        )
        input_ids = batch["input_ids"].cuda()  
        attention_mask = batch["attention_mask"].cuda()
        
        logits_dense = torch.softmax(qwen3(input_ids, attention_mask).logits / 0.6, dim=-1, dtype=torch.float32)
        logits_sparse =torch.softmax(qwen3_nsa(input_ids, attention_mask).logits / 0.6, dim=-1, dtype=torch.float32)
        acc_rates = torch.minimum(logits_sparse, logits_dense).sum(dim=-1)
        
        acc_rates = (acc_rates * attention_mask).sum() / attention_mask.sum().clamp_min(1)

        print(acc_rates) 
