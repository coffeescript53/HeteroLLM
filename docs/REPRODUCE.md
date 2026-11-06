# Reproduction guide

The accepted study is preserved as evidence. This guide separates the runnable public CPU checks from historical runtime provisioning and GPU reproduction. The November 3 exploratory quality attempt stopped at its GPU temperature limit; it produced no valid E/P quality comparison. See [current status](STATUS_REPORT.md). Historical execution manifests remain immutable; completed output paths and old source bindings must never be recycled.

## 1. Inspect the package without a model or GPU

The Python package requires Python 3.10+ and uses the standard library. Start with the bundled measurement replay; it needs no model, GPU, network access, or installation:

```bash
python3 scripts/demo.py
bash scripts/test-offline.sh
```

The broader host-supervision suite additionally requires Linux process APIs, readable procfs, and memory-pressure telemetry at `/proc/pressure/memory`:

```bash
bash scripts/test-cpu.sh
```

This entry point runs CLI help, the selected host regression modules, and the C3 synthetic trace-analysis tests. Tests use synthetic backends, temporary files and owned CPU subprocesses. Linux process/signal and `/proc` semantics are required for supervisor tests. It does not load weights, inspect physical GPUs or run inference. One comparison against the private frozen archive is explicitly skipped because that archive is not distributed. This is a selected Linux host subset, not the complete research-host regression suite. Containers without PSI should use the offline subset; see [verification](VERIFICATION.md). Preserve the executable bit on `tests/fixtures/fake_runtime.py` when copying files; it is a Python protocol fixture, not a model runtime.

The Stage B replay fixtures are C4 privacy-sanitized derivatives: local paths were replaced before the public export. Public tests verify their own exact hashes; comments retain the different original archived hashes. The loader line and response content are preserved. These public files are not byte-identical substitutes for the private original evidence.

With an existing C++17 compiler available as `c++`, the standalone v2 recorder tests are also CPU-only:

```bash
bash scripts/test-recorder-cpu.sh
```

These seven tests compile three tiny fixtures sequentially, with 30-second compiler and 10-second fixture limits. They verify 100,000 concurrent event calls, retained copy/graph details, exact aggregates, loss rejection and disabled behavior. The separate patch-reconstruction test needs omitted pinned upstream source trees and is not run by this entry point. Neither entry point installs dependencies.

To inspect the planner without device discovery, supply a synthetic snapshot explicitly:

```bash
snapshot=$(mktemp)
PYTHONPATH=tests python3 -c 'import json; from helpers import snapshot; print(json.dumps(snapshot()))' > "$snapshot"
python3 -m heterollm preview --config configs/llama33-70b.json --snapshot "$snapshot"
rm -- "$snapshot"
```

The output is a dry plan, not measured performance or permission to execute. Without `--snapshot`, planner commands can query installed GPU tools. The [offline demo](DEMO.md) needs no physical devices.

An optional virtual environment can expose the console entry point:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
heterollm --help
```

Installation may download build tooling if unavailable; it is not needed to read or run the package from its source directory. No package installation is part of the public CPU check. The committed SVG/PNG charts need no plotting dependency to view. Rendering new chart derivatives requires Matplotlib; inspect the chart data/provenance first and write derivatives to a new output directory.

### Other CPU tests have explicit prerequisites

The complete research-host command is `python3 -m unittest discover -s tests -v`. Its successful historical results are host results, not a promise of a dependency-free fresh-machine run. Recorder tests compile C++17 fixtures. Several fake-backend tests additionally use the exact pinned upstream headers/source and link the existing `libggml-base`. Contract tests reference frozen local configs, reference evidence and runtime receipts that are deliberately absent from the public export. No GPU is exercised by these tests, but “CPU-only” does not mean “independent of the runtime source/build.”

Do not rewrite historic identity fixtures or skip failing safety assertions to obtain a green result on another machine. Report which prerequisite-dependent checks are unavailable. A future port needs separate contract fixtures and review; this portfolio does not claim that port was completed.

## 2. Recorded runtime build — future reproduction only

The study used one original mixed CUDA/HIP runtime for all C1 configurations and separate isolated instrumented runtimes for C2. Model loading, kernels, layer placement and scheduling are upstream llama.cpp/GGML functionality.

| Item | Recorded value |
| --- | --- |
| Source revision | `5e03bdd8700948b9c41c54dd1b00f28a2aebc03f` |
| C/C++ compiler | GNU 13.3.0 |
| CUDA toolkit/compiler | 13.2.86 |
| HIP compiler | Clang 23.0.0, recorded ROCm 10.0 installation |
| Build | CMake + Ninja, Release, shared backends |
| CUDA architecture | `120a-real` |
| AMD target | `gfx1201` |
| Bounded compilation | Two jobs with host thermal/resource monitoring |

[BUILD_OPTIONS.txt](BUILD_OPTIONS.txt) contains the normalized cache controls from the frozen original build: boolean/string GGML/LLAMA options and the selected build/target controls. Compiler paths are supplied separately because installation paths differ. It is a reproducible configuration recipe, not a claim of bit-identical output across toolchains, operating systems or dependencies. Preserve a new cache, source state, compiler versions, resolved libraries and hashes of executable/backend libraries after any separately approved build.

The following is a **future build recipe**, not a command run during C4. Provision compatible drivers/toolchains according to the pinned upstream documentation; this project does not install or upgrade system software. Define `CUDA_ROOT`, `ROCM_PATH` (SDK discovery root), and `ROCM_COMPILER_ROOT` (compiler/library installation) for that installation. The recorded SDK discovery root and compiler root were distinct; do not assume they are interchangeable. Run the build only with the project's accepted CPU/DIMM and memory monitoring in place; the historical bounded build scripts document that policy but have used output paths and are not fresh launchers.

```bash
# Caller supplies the three installation roots above.
export PATH="$CUDA_ROOT/bin:$ROCM_PATH/bin:/usr/local/bin:/usr/bin:/bin"
export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_COMPILER_ROOT/lib"
export LANG=C.UTF-8 LC_ALL=C.UTF-8 CMAKE_BUILD_PARALLEL_LEVEL=2
# Original uninstrumented runtime in a new destination.
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
git -C llama.cpp checkout --detach 5e03bdd8700948b9c41c54dd1b00f28a2aebc03f
mapfile -t build_options < <(sed '/^#/d; /^$/d; s/^/-D/' docs/BUILD_OPTIONS.txt)
cmake -S llama.cpp -B llama.cpp/build-mixed -G Ninja \
  "${build_options[@]}" \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCMAKE_HIP_COMPILER="$ROCM_COMPILER_ROOT/lib/llvm/bin/clang"
cmake --build llama.cpp/build-mixed --parallel 2 --target llama-bench llama-cli
```

The existing workstation runtime must not be reused as this new build destination. For C2 v3 reproduction, a separate fresh checkout of the same revision receives `instrumentation/v3/phase-timing.patch`; this full patch already incorporates v2 CPU recording. Do not apply v1, v2 and v3 patches cumulatively. Configure with the same normalized options and additionally `-DCMAKE_CXX_FLAGS=-DHETEROLLM_CPU_TRACE=1`. Rebuild only in that isolated tree. Validate the exact patch, source and library resolution against the new receipt; a process must not accidentally load the original uninstrumented GGML DSOs.

Trace-enabled T uses both `HETEROLLM_TRACE_FILE` and `HETEROLLM_TIMING_FILE` with fresh absolute output files. U is the original uninstrumented binary with those tracing variables absent. A compiled instrumented binary with tracing disabled is a distinct arm; it must not be silently substituted for U. See the [instrumentation overview](../instrumentation/README.md).

## 3. Model identity and acquisition boundary

The model is not distributed. The accepted quantized file is:

```text
Repository: bartowski/Llama-3.3-70B-Instruct-GGUF
Revision:   b6c5c9f176f3279204034e1d16d393105e95cb88
Filename:   Llama-3.3-70B-Instruct-Q4_K_M.gguf
Size:       42520398816 bytes
SHA-256:    32df3baccb556f9840059b2528b2dee4d3d516b24afdfb9d0c56ff5f63e3a664
```

Review the original model license and use terms linked in [NOTICE.md](../NOTICE.md). `scripts/acquire-model.sh` without arguments prints the pinned acquisition command; it does not download. Actual acquisition and the complete checksum/GGUF directory read are separate resource-consuming operations, not needed for the public CPU demo and not authorized by this guide.

The study's launch checks reuse a prior full checksum receipt plus size/path/mtime checks. That is a transparent reproducibility tradeoff, not protection against adversarial metadata-preserving replacement. Another machine needs its own complete integrity receipt.

## 4. Exact recorded C1 workload specification

These are the four quiet child argv specifications, shown with neutral `BENCH`/`MODEL` placeholders. They document what was measured. **Do not invoke them directly as an alternative to supervised execution.** The supervisor supplies controlled environment, fresh identity/capacity/sensor checks, thermal admission, deadlines, evidence and cleanup.

```bash
common=(
  -m "$MODEL" -sm layer -fitt 0 -p 512 -n 128 -d 0
  -b 512 -ub 128 -t 16 -ctk f16 -ctv f16 -fa off
  -nkvo 0 -nopo 0 --no-host 0 --repack 1 -lm none -lzm off
  -r 3 -o json --offline -fitc 0 -embd 0 -ncmoe 0 -mg 0
  --prio 0 --delay 0 --poll 50 --cpu-strict 0 -C 0x0
)
# A — CUDA0 + RAM
"$BENCH" "${common[@]}" -dev CUDA0 -ngl 56
# B — ROCm0 + RAM
"$BENCH" "${common[@]}" -dev ROCm0 -ngl 59
# D — default heterogeneous
"$BENCH" "${common[@]}" -dev CUDA0/ROCm0 -ngl 999
# E — equal heterogeneous
"$BENCH" "${common[@]}" -dev CUDA0/ROCm0 -ngl 999 -ts 1/1
```

The actual suite order was `E D A B → D B E A → B A D E`, twelve sequential new processes. Each process measures PP512 and depth-zero TG128 separately with three repetitions. The controlled child environment included `LANG=en_US.UTF-8`, `LC_ALL=C.UTF-8`, a reviewed tool search path and the required per-process ROCm library search path. `GGML_SCHED_DEBUG` was absent and `-v` was absent during C1 timing. Preserve the exact environment/library resolution when reproducing; do not make a global linker or power-setting change.

The historical `heterollm.stage_c1` adapter is the authoritative supervised workflow. A fresh execution requires a new output directory, correct physical mapping, new approved source/runtime/hardware contract and explicit authorization. The old completed C1 manifest cannot be made fresh by changing a date or ignoring its source mismatch. Do not use the generic `tune --execute` path: that is a different search, not the accepted C1 design.

Mandatory current controls include 900 seconds per process, the reviewed suite deadline, 40 GiB available-RAM admission, 32 GiB child RSS/HWM ceiling, 4 GiB available-RAM floor, 256 MiB cumulative swap-growth stop, 85°C GPU runtime stop and mandatory CPU/DIMM/GPU monitoring. GPU cooldown is at least 30 seconds and then both GPUs at/below 45°C for ten seconds across eleven fresh readings, with a 600-second maximum gate. Capacity screening may refuse the tight single-GPU placements; do not reduce layers or loosen thresholds to force a pass. See [methodology](METHODOLOGY.md) for distinctions between safeguards present during C1 and later CPU/DIMM protections.

## 5. Reproduce analysis, not a new experiment

Use the public launch-level measurements and plotting code to recompute C1 means/sample SDs and visualizations. Keep original integer repetition durations in the provenance extract. For overhead, compare paired process mean elapsed times using the six predeclared UT/TU blocks, not C1 history or pooled internal repetitions. Keep numerical-equivalence results scoped to the fixed teacher-forced fixture. The existing `heterollm.trace_analysis`, `heterollm.timing_analysis` and `heterollm.calibration_analysis` modules perform offline validation/analysis; imports do not launch a model.

Missing raw trace or private contract data is unavailable evidence, not a zero or a passed check. Independent reproduction of the full research archive requires its original files, permissions and hashes. The compact public dataset is a reviewable derivative, not a replacement for the immutable archives.

The remaining publication steps are listed in [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md). No additional GPU run is needed to present the baseline and measurement-engineering portfolio. C5 has not established quality preservation or a controlled performance improvement; its original failed validation and later temperature-limited attempt remain failures. The 12-process E/P performance comparison remains blocked. The [C5 proposal](C5_PLACEMENT_PROPOSAL.md) is historical context, not an execution contract; [current status](STATUS_REPORT.md) takes precedence.
