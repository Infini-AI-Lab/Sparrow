#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launcher_name="$(basename "$0")"

cd "$script_dir"

for script in *.sh; do
  if [ "$script" = "$launcher_name" ]; then
    continue
  fi

  EXP="${script%.sh}"
  echo "Submitting $script as job $EXP"
  sbatch --job-name="$EXP" "$script" "$EXP" ""
done
