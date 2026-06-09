policy="qwen3-14b-0.86" 
# policy="qwen3-4b-0.75" 
python examples/verify_algooldanother.py \
  --model-name Qwen/Qwen3-14B \
  --trials 32 \
  --topk-val 61 \
  --block-size 16 \
  --page-size 16 \
  --vortex-module-name full_attention \
  --data-path examples/aime24.jsonl \
  --mem 0.8 \
  --policy-name "${policy}" \
  --generation-max-new-tokens 32768 
