# Engineering journey: making the measurements trustworthy

Research through 3 November 2026. This is a current retrospective; original run records, failed outcomes and dated reports remain unchanged. The study uses Llama 3.3 70B Q4_K_M, an RTX 5090, a Radeon AI PRO R9700 and 96 GB host RAM. Earlier memory configurations and other models are historical context, not additional observations in the controlled baseline.

The project began with a capacity question: could two different 32 GB GPU backends serve a model larger than either device without substantial CPU offload? llama.cpp already supplied mixed-backend execution. The engineering task was to establish what actually ran, make failures durable, and obtain comparisons whose controls and limitations could be inspected.

## Loading and admission failures

| Failure or misleading signal | Evidence and correction | What remains uncertain |
|---|---|---|
| The original 48 GB attempt stopped on swap growth before model buffers were reported. | The pinned loader requested whole-file `MAP_POPULATE` before main backend-buffer creation with `mmap/off`. The next reviewed experiment selected supported `none/off` loading and added process-specific memory and durable cleanup evidence. Successful experiments subsequently used 96 GB. | The original samples do not identify which anonymous pages were swapped or establish the precise kernel/driver cause. The hardware and loading changes prevent treating later success as a controlled test of either change alone. |
| Final GPU admission lost the AMD temperature after an idle interval. | A sensor-only reproduction observed active → suspended transitions and recovered fresh readings through a bounded, physical-device-verified render-node open. Error records now preserve missing fields, raw readings and power state. | The first failed record discarded its sensor detail, so autosuspend is strongly consistent with that incident rather than retrospectively proven. No stale temperature is substituted; failed recovery stops admission. |
| A guard read `none)` as an unauthorized loading mode. | Parsing changed from a loose token regex to complete loader records, including the real `(load_mode = none)` syntax. Saved stderr was replayed at deliberately split read boundaries; unauthorized and malformed modes still fail. | A parser pass establishes the reported loading mode, not model correctness. Incomplete live records are deferred, not accepted as valid fields. |
| Functional output validation disagreed with a completed response. | Assistant-response extraction and validation were corrected using saved output, separating model response from surrounding transcript/log text. The original run outcome remained frozen. | An arithmetic response is a small functional check, not a general correctness evaluation. Later fixed-input full-logit comparison provides a different, explicitly limited test. |

The loading source references are [mapping initialization](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/src/llama-model.cpp#L1814), [mapping construction](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/src/llama-model-loader.cpp#L1410), and [Linux population flags](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/src/llama-mmap.cpp#L482). These establish requested source behavior; they do not reconstruct missing process measurements.

## Benchmark orchestration needed engineering too

The first successful measurements did not excuse defects in the surrounding controller. Each correction preserved real resource stops and left previous evidence intact.

- **Cooldown scheduling:** an AMD reading of 50°C exceeded the 45°C admission criterion after a fixed 30-second delay. This was an admission scheduling failure, not evidence of runtime overheating. A bounded thermal gate retained the 30-second minimum, required both GPUs at or below 45°C for ten seconds across eleven fresh readings, and stopped after 600 seconds or the suite deadline. The 85°C runtime stop remained unchanged.
- **Evidence-accounting race:** a monitor's atomic temporary file could disappear between directory enumeration and `stat`. A shared helper tolerates disappearance only for recognized transient atomic-write files. Permanent-file failures still propagate; oversized evidence still stops the experiment. Deterministic disappearance and concurrent-writer tests protect both supervisor and cooldown paths.
- **Post-reboot identity:** Linux `MemTotal` changed by 12 KiB while the other recorded identities and controls matched. Both identity checks gained a ±1 MiB allowance against the original baseline, with original/observed values, signed difference and decision recorded. The 40 GiB available-RAM admission requirement and every other identity comparison remained unchanged.
- **Durable termination:** the supervisor owns the workload group, records stop intent before signaling, escalates bounded TERM/KILL cleanup, and saves exit/resource evidence from the surviving parent. Monitoring errors and unexpected exceptions are failures, not missing data that can be ignored.

The completed C1 comparison contains twelve fresh independent processes. Earlier partial suites are excluded. Three repetitions inside a process remain correlated; the statistical unit is the process. C0 diagnostics establish workload and placement behavior, but their timings do not become formal C1 observations.

## Instrumentation that can reject its own output

The first routing recorder completed both workloads but exhausted 65,536 detailed slots and dropped another 22,336 spans. Its retained records were dominated by repetitive event bookkeeping, with no retained copy or graph submissions. The diagnostic correctly failed. A missing copy record in that incomplete capture was never interpreted as zero transfers.

Recorder v2 kept detailed routing/copy/graph spans and represented frequent event calls with exact bounded per-backend counts, total/max CPU duration and duration histograms. Intentional aggregation is distinct from unexpected loss. Overflow, incomplete scopes, malformed counters, errors and write failures invalidate the trace. CPU fake-backend tests checked copied data and callback order under high event traffic, not only compilation.

The accepted follow-up retained **5,222 detailed spans and 82,650 aggregate calls, with zero unexpected losses**. It captured 137 host-staging parent spans carrying 37,781,504 logical bytes. These observed counts demonstrate the recorder repair; they do not measure physical bus traffic or an inference speedup.

The v3 extension added warmup/PP/TG attribution and completed same-device CUDA/HIP event intervals. Completion is checked after existing waits; the instrumentation does not insert waits simply to force readable timing. Cross-vendor ordering uses host-side bounds rather than subtracting independent device timestamps. Stream residence remains different from kernel execution, and nested CPU intervals are analyzed as unions rather than summed as separate costs.

Two controlled validation steps made this useful evidence:

1. Original and instrumented mixed-vendor runtimes produced byte-identical captures for 129 full-vocabulary float32 vectors: 16,545,024 values, with matching greedy token IDs and decoded token-piece bytes. This is fixed-fixture agreement, not arbitrary-generation equivalence.
2. Six contemporaneous counterbalanced U/T process pairs measured observer overhead. PP512 was +0.108100% and TG128 +0.605006%, with nominal 95% intervals reported in the [portable evidence](../evidence/portfolio/). The calibration applies to this v3 observer and workload, not future profilers.

CPU/DIMM monitoring was added after C1. C1 therefore has no CPU/DIMM thermal history; later telemetry must not be projected backward onto it. DDR5 SPD-hub thresholds are provisional engineering safeguards, and warning alarms are distinct from critical alarms.

## Stopping at a defensible boundary

C3 found a longer ROCm graph region on the sequential TG path. It also showed that most time inside the scheduler's fallback scope was nested producer synchronization, not the staged-copy parent call. Neither observation identifies removable work. C3.1 checked applicable public kernels, dispatch changes, graph mechanisms and scheduler work before claiming novelty.

C3.2 ended with a source-level **NO-GO** for the proposed ROCprofv3 route under the unchanged completeness and resource contract. Extracted-source CPU counterexamples found loss-accounting and blocking problems below the tool callback; bounded tool-only changes could not establish the required guarantees. This is a scoped source/CPU finding, not a live GPU reproduction or a universal claim about every profiler version. That route remains closed.

## A promising placement is not a completed optimization

C5 tested the existing `--tensor-split` control with a CUDA-heavier P placement. Its capacity pilot passed and recorded 853.196306 PP512 and 20.286076 TG128 tokens/s. Comparison with older E measurements was only a screening signal. The next strict fixed-input E/P test failed: 16,448,640 values exceeded the frozen elementwise tolerance, and 128 of 129 greedy predictions agreed. A high logit failure count is not a task-accuracy score; it also cannot be dismissed as harmless without additional evidence. The original failure was preserved.

Allocation validation exposed a separate mistake: a historical median-based matcher was too rigid to serve as a universal gate across different workload graphs. P's sampled peak did not exceed its earlier pilot peak, but the validator still rejected its allocation evidence. That does not resolve the cause of the changed memory pattern. Later quality preparation separated conservative capacity and live safety gates from workload-specific observed compute buffers, without retroactively passing the original run.

R2 then stopped before model execution because a reboot changed DIMM I²C numbering. The narrow fix accepted enumeration changes only under the recorded controller, adapter, addresses and module-part checks; missing or changed identities still failed. R3 passed that admission and loaded E. The active AMD junction reading reached **86°C**, triggering the unchanged **85°C** hottest-channel stop. The fan read 1,214 RPM 0.096 seconds before that sample; no stall was recorded. P never launched, partial output was retained and all owned processes were cleaned up. The thermal cause and exploratory E/P quality remain unresolved.

The [current optimization decision](OPTIMIZATION_DECISION.md) is unchanged: no specific source-level optimization is yet proven or ready to implement. C4 presents the completed measurement engineering; C5 remains unfinished. A failed candidate does not prove all remaining opportunities exhausted, and a public portfolio does not need an invented speedup.
