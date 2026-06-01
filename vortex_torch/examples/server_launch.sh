export OPENAI_API_KEY="None"
MODEL_NAME=$1
TP_SIZE=$2
python -m sglang.launch_server \
 --model-path "$MODEL_NAME" \
 --page-size 64 \
 --disable-overlap-schedule \
 --attention-backend "flashinfer" \
 --vortex-layers-skip 0 \
 --vortex-block-reserved-eos 1 \
 --vortex-block-reserved-bos 2 \
 --vortex-topk-val 29 \
 --vortex-block-size 16 \
 --vortex-workload-chunk-size 64 \
 --vortex-module-name "block_sparse_attention" \
 --vortex-max-seq-lens 32768 \
 --context-length 32768 \
 --mem-fraction-static 0.85 \
 --vortex-compilation-cache-dir "~/.vortex_compilation_cache" \
 --enable-vortex-sparsity \
 --tp-size "$TP_SIZE" \
 --port 30000 \
 --host 127.0.0.1