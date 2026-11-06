"""One bounded synthetic CPU cost probe; never loads GPU runtime or any model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

SOURCE = r'''
#include "heterollm-cpu-trace.h"
#include <chrono>
#include <cstdio>
#include <cstdint>
static uint64_t clock_ns() {
 return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
int main() {
 uint64_t start = clock_ns(), initialized = 0, loop_end = 0;
 volatile uint64_t checksum = 0;
 {
#if defined(HETEROLLM_CPU_TRACE) && HETEROLLM_CPU_TRACE
  heterollm_trace::Recorder recorder;
#endif
  initialized = clock_ns();
  for (uint64_t i = 0; i < 10000; ++i) {
#if defined(HETEROLLM_CPU_TRACE) && HETEROLLM_CPU_TRACE
   size_t slot = recorder.begin("synthetic.noop", "CUDA0", "ROCm0", 8192);
#endif
   checksum += i;
#if defined(HETEROLLM_CPU_TRACE) && HETEROLLM_CPU_TRACE
   recorder.end(slot);
#endif
  }
  loop_end = clock_ns();
 }
 uint64_t flushed = clock_ns();
 std::printf("{\"init_ns\":%llu,\"loop_ns\":%llu,\"flush_ns\":%llu,\"iterations\":10000,\"checksum\":%llu}\n",
  (unsigned long long)(initialized-start), (unsigned long long)(loop_end-initialized),
  (unsigned long long)(flushed-loop_end), (unsigned long long)checksum);
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("fresh output required")
    records = {}
    with tempfile.TemporaryDirectory(prefix="heterollm-cpu-trace-cost-") as temp:
        directory = Path(temp)
        source = directory / "cost.cpp"
        source.write_text(SOURCE)
        for name, flags in [("compiled_disabled", []), ("enabled", ["-DHETEROLLM_CPU_TRACE=1"])]:
            binary = directory / name
            subprocess.run(["c++", "-std=c++11", "-O2", "-pthread", *flags,
                            "-I", str(Path(__file__).resolve().parent), str(source), "-o", str(binary)],
                           check=True, capture_output=True, timeout=30)
            output = directory / (name + ".jsonl")
            began = time.monotonic_ns()
            result = subprocess.run([str(binary)], env=dict(os.environ, HETEROLLM_TRACE_FILE=str(output)),
                                    text=True, capture_output=True, check=True, timeout=10)
            records[name] = json.loads(result.stdout)
            records[name].update(process_wall_ns=time.monotonic_ns() - began, stderr=result.stderr,
                                 trace_size_bytes=output.stat().st_size if output.exists() else 0)
            if name == "enabled":
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                if not rows[-1]["complete"] or rows[-1]["recorded"] != 10000 or result.stderr:
                    raise SystemExit("synthetic trace failed integrity")
                records[name]["trace_footer"] = rows[-1]
                records[name]["record_bytes"] = rows[0]["record_bytes"]
    records["interpretation"] = (
        "Single synthetic CPU sample of 10,000 no-op recorder begin/end pairs, not GPU inference, "
        "not a statistical overhead estimate. Compile-disabled comparison retains identical volatile checksum loop. "
        "Enabled initialization allocates the full bounded record pool and durably writes a header. "
        "Flush includes JSON serialization, fflush, fsync and close; reported separately. "
        "Real workload overhead, contention and GPU scheduling perturbation remain unmeasured.")
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
