#!/bin/sh
# Optional standalone C++17 recorder fixtures; no CUDA, HIP or model runtime.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/tests:$PWD"
command -v c++ >/dev/null
"$PYTHON" -m unittest -v \
 test_c2_trace_v2.RecorderV2Tests.test_100000_concurrent_aggregates_preserve_all_interleaved_details_exactly \
 test_c2_trace_v2.RecorderV2Tests.test_zero_host_stage_complete_trace_is_valid_and_waits_not_double_counted \
 test_c2_trace_v2.RecorderV2Tests.test_loss_unfinished_malformed_and_counter_overflow_fail_closed \
 test_c2_trace_v2.RecorderV2Tests.test_disabled_arguments_and_output_unchanged \
 test_c2_trace_v2.RecorderV2Tests.test_abrupt_exit_and_existing_output_refused \
 test_c2_trace_v2.RecorderV2Tests.test_analyzer_rejects_bad_schema_counters_histograms_and_metadata \
 test_c2_trace_v2.RecorderV2Tests.test_histogram_total_must_include_one_call_at_the_reported_maximum
