#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
repo=bartowski/Llama-3.3-70B-Instruct-GGUF
revision=b6c5c9f176f3279204034e1d16d393105e95cb88
filename=Llama-3.3-70B-Instruct-Q4_K_M.gguf
if [[ "${1:-}" != --execute ]]; then
  echo 'Preview only. Requires the owner’s approved download plan.'
  printf 'HF_HUB_DISABLE_XET=1 timeout --signal=TERM --kill-after=10s 7200 hf download %q %q --revision %q --local-dir models/llama33-70b --max-workers 1\n' "$repo" "$filename" "$revision"
  exit 0
fi
python3 - <<'PY'
import shutil
if shutil.disk_usage('.').free < 90_000_000_000:
    raise SystemExit('Require 90 GB free disk for download, partial data, and safety reserve')
PY
HF_HUB_DISABLE_XET=1 timeout --signal=TERM --kill-after=10s 7200 \
  hf download "$repo" "$filename" --revision "$revision" \
  --local-dir models/llama33-70b --max-workers 1
# Integrity receipt is deliberately a separate, explicit command in the approval plan.
