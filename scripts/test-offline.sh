#!/bin/sh
# Public analysis fixtures; no host telemetry, device discovery, or model.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/tests:$PWD"
"$PYTHON" -m unittest -v test_c2_trace_analysis test_portfolio_data
"$PYTHON" -m unittest discover -s analysis/archived/c3 -p 'test_*.py' -v
