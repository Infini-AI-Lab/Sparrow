#!/usr/bin/env bash

# seqlens=(24000 32000) 
# enableits=(0 1) 
# page_sizes=(8 16 32 64) 
# tokenlevelbudgets=(256 512 1024 2048) 

# for enableit in ${enableits[@]}; do 
#     for seqlen in ${seqlens[@]}; do 
#         batch_size=16 
#         if [ ${enableit} -eq 0 ]; then 
#             python profile_100_decode.py \
#                 --model-path Qwen/Qwen3-1.7B \
#                 --prefill-chunk-size 512 \
#                 --vortex-algorithm BLOCK_TOPK \
#                 --vortex-num-selected-pages 29 \
#                 --page-siz3 16 \
#                 --enableit 0 \
#                 --sequence-length mean \
#                 --decodingsteps ${seqlen} \
#                 --batch-size ${batch_size} | tee -a 2b_gridlatency_decode5.txt 
#         else 
#             for page_size in ${page_sizes[@]}; do 
#                 for budget in ${tokenlevelbudgets[@]}; do 
#                     numselectedpages=$((${budget} / ${page_size})) 
#                     python profile_100_decode.py \
#                         --model-path Qwen/Qwen3-1.7B \
#                         --prefill-chunk-size 512 \
#                         --vortex-algorithm BLOCK_TOPK \
#                         --vortex-num-selected-pages ${numselectedpages} \
#                         --page-siz3 ${page_size} \
#                         --enableit ${enableit} \
#                         --sequence-length mean \
#                         --decodingsteps ${seqlen} \
#                         --batch-size ${batch_size} | tee -a 2b_gridlatency_decode5.txt 
#                 done 
#             done 
#         fi 
#     done 
# done 

pagesizes=(16 32 64 128) 
for pagesize in ${pagesizes[@]}; do 
    numpage=$((1024 / $pagesize - 3)) 

    CUDA_VISIBLE_DEVICES=0 python profile_100_decode.py \
        --model-path Qwen/Qwen3-1.7B \
        --prefill-chunk-size 128 \
        --vortex-algorithm "block_sparse_attention" \
        --page-siz3 ${pagesize} \
        --vortex-topk-val ${numpage} \
        --enableit 1 \
        --sequence-length "16000" \
        --decodingsteps 2000 \
        --batch-size 16 
done 
