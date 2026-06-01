#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
template_file="${script_dir}/verify_algo.sh"
output_dir="${script_dir}/upscrip"

# policies=(
#   "qwen3-1.7b-0.86"
#   "qwen3-1.7b-0.88"
#   "qwen3-1.7b-0.90"
#   "qwen3-1.7b-0.92"
#   "qwen3-1.7b-0.94" 
#   "qwen3-1.7b-0.75"
#   "qwen3-1.7b-0.80" 
# ) 
policies=(
  "qwen3-4b-0.75"
  "qwen3-4b-0.80"
  "qwen3-4b-0.86"
  "qwen3-4b-0.88"
  "qwen3-4b-0.90"
  "qwen3-4b-0.92" 
) 

mkdir -p "${output_dir}"

for policy in "${policies[@]}"; do
  output_file="${output_dir}/verify_algo_${policy}.sh"

  sed "s/^policy=.*/policy=\"${policy}\" /" "${template_file}" > "${output_file}"
  chmod +x "${output_file}"

  echo "Generated ${output_file}"
done 
