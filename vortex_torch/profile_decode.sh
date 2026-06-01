nsys profile \
  -o q_2b_full \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite true \
  python profile_decode.py \
    --model-path Qwen/Qwen3-1.7B \
    --batch-size 4 \
    --max-new-tokens 64 \
    --input-len 12288
