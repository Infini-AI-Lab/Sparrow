#!/bin/bash
#SBATCH --job-name=verl_ray_job
#SBATCH --output=/home/xun/yangzho6/distilldynamsparse/vortex_torch/scripts_two/logs/%x_%j.out 
#SBATCH --error=/home/xun/yangzho6/distilldynamsparse/vortex_torch/scripts_two/logs/%x_%j.err 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=6:00:00
#SBATCH -p medium 
#SBATCH --exclude=compute-node-40 

set -x 

# source ~/miniconda3/etc/profile.d/conda.sh 
# conda activate distillsparsetwo 

which python 
which python3 
which pip
python -m pip --version 

#!/usr/bin/env bash 
# export PYTHONPATH=/workspace/distilldynamsparse/vortex_torch:$PYTHONPATH 
export PYTHONPATH=/workspace/distilldynamsparse2/sglang/python:$PYTHONPATH 

cd /workspace/distilldynamsparse2/vortex_torch 

set -x 

policies=("qwen3-1.7b-0.86" "qwen3-1.7b-0.88" "qwen3-1.7b-0.90" "qwen3-1.7b-0.92" "qwen3-1.7b-0.94") 

# for policy in "${policies[@]}"; do 

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
