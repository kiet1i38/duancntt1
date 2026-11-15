#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo 'Usage: bash experiments/review-20261113/run_gpu.sh OUTPUT_ROOT DEADLINE_UTC_UNIX' >&2
  exit 2
fi
output_root="$1"
deadline="$2"
for model in qwen2.5-7b qwen2.5-3b; do
  python -B experiments/review-20261113/gpu_runner.py --model "$model" \
    --output "$output_root/$model" --batch-size 8 --deadline-utc "$deadline"
  python - "$output_root/$model/manifest.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf8') as stream:
    result = json.load(stream)
if result['status'] != 'complete':
    raise SystemExit('Stopped at checkpoint; remaining models have not run.')
PY
done
