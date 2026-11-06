#!/bin/sh
# Linux host checks: fixtures and owned CPU subprocesses only.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/tests:$PWD"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"; assert sys.platform == "linux", "Linux process and /proc semantics required"'
if [ ! -r /proc/pressure/memory ]; then
    echo "Host prerequisite unavailable: /proc/pressure/memory (Linux PSI)." >&2
    echo "The supervisor requires real telemetry. Run scripts/test-offline.sh for analysis fixtures." >&2
    exit 2
fi
"$PYTHON" -m heterollm --help
"$PYTHON" -m unittest -v \
  test_core test_runner test_supervisor test_gpu_sensors \
  test_host_thermal test_evidence_sizes test_memory_identity \
  test_loading_mode test_assistant_response \
  test_c2_trace_analysis test_portfolio_data
"$PYTHON" -m unittest discover -s analysis/archived/c3 -p 'test_*.py' -v
