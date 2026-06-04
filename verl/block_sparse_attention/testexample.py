from transformers import AutoModelForCausalLM, AutoTokenizer 

import verl.block_sparse_attention.bsa_backend 
import torch 

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base") 

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", device_map = "cuda:0") 
model.config._attn_implementation = "nsa_triton" 

batch_size = 6 
seq_len = 4024 
input_ids = torch.randint((batch_size, seq_len), max_value = 100000) 

output = model(input_ids=input_ids) 
print("the pass is done. ") 
