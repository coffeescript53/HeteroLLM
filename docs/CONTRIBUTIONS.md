# Contributions and attribution

HeteroLLM evaluates dense 70B inference across two GPU vendors and makes the experiment's controls, failures and measurement limits inspectable. Its strongest demonstrated original change is a bounded trace-recorder repair: preserve important routing evidence under heavy event traffic, then validate the observer's numerical behavior and overhead. llama.cpp supplies inference, placement and kernels.

The first recorder filled 65,536 slots with repetitive event calls and dropped 22,336 additional spans. The revised recorder retained 5,222 detailed spans and exactly aggregated 82,650 event calls with zero unexpected losses. The accepted capture exposed 137 host-staged copies carrying 37,781,504 logical tensor bytes. This is an instrumentation reliability result, not an inference speedup or a physical PCIe traffic measurement.

## What came from upstream

The pinned llama.cpp/GGML runtime supplies GGUF model loading, transformer graph construction, quantized kernels, CUDA and HIP backends, layer assignment, backend buffers, graph replay, scheduling and generic copy fallback. NVIDIA and AMD provide their runtime/toolchain interfaces. The model and its quantization are external artifacts, not HeteroLLM training or model-development contributions.

The pinned [scheduler](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-backend.cpp#L1938-L1948) and [copy implementation](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-backend.cpp#L628-L649) already implement the path observed in this study. Layer execution on both CUDA and ROCm is validated upstream functionality, not an original replacement scheduler.

## What this project implemented

| Project contribution | Demonstrable artifact or outcome | Scope |
|---|---|---|
| Hardware-aware capacity screening | GGUF tensor-directory accounting, layer/output grouping, resource estimates and explicit rejection reasons | Estimates screen candidates; actual runtime allocation evidence remains necessary. |
| Bounded experiment supervision | Exact source/runtime/dependency contracts, physical GPU identity, mandatory sensors, thermal/resource/deadline stops, gated release and process-group cleanup | Host-tested safety engineering, not a guarantee against kernel failure, power loss or unstable hardware. |
| Controlled comparison | Twelve independent C1 launches across NVIDIA + RAM, AMD + RAM, default mixed and equal mixed; retained raw samples and launch-level statistics | Four fixed configurations, one machine/model; no placement sweep or selected optimum. |
| Routing recorder v2 | Detailed routing/copy/graph spans plus exact bounded event-call aggregates; strict overflow/completeness validation | CPU-call evidence; not native kernel activity or physical traffic. |
| Phase/completed-device timing v3 | Warmup/PP/TG markers, completed same-device native event pairs and CPU correlation bounds | Stream residence can include waits and gaps; cross-vendor clocks are not subtracted. |
| Correctness validation | Deterministic full-vocabulary U/T captures with finite/size/hash/position/token checks | All 16,545,024 values and 129 predictions/pieces matched for the fixed fixture; not universal numerical equivalence. |
| Observer calibration | Six paired U/T blocks, counterbalanced order, process-level elapsed ratios, uncertainty and order strata | Estimates the accepted observer's cost on matched PP512/TG128 workloads. |
| Evidence-led negative decisions | Nested-interval attribution, source applicability audit and a bounded profiler NO-GO | Does not establish a causal dominant bottleneck, a new kernel optimization or a speedup. |

The source includes an earlier bounded candidate-planning/tuning path that was CPU-tested. The accepted GPU study used separately reviewed fixed-stage adapters. Do not present the unused search scaffolding as an empirically validated optimizer or claim that it selected the fastest placement.

## Review and authorship transparency

This was an AI-assisted engineering workflow. The project owner set the research goals, controlled hardware, reviewed proposals and separately authorized experiments. Codex assisted with implementation, testing, execution and evidence preparation. ChatGPT performed independent offline reviews of specified source/evidence packages. Independent here means a separate review of saved artifacts; it is not an independent physical rerun, external laboratory certification or human peer review.

C1/C2 findings, C3 attribution and C3.1 prior-art conclusions received independent artifact review within their recorded scopes. C3.2 is a completed source/CPU NO-GO, not a live GPU reproduction of every identified profiler issue. Since 3 November 2026, the workflow is owner + Codex; the external reviewer role has ended. Later Codex checks are not independently verified results. A package hash proves byte integrity, not scientific correctness.

The owner should use first-person resume claims only for responsibilities they can explain and defend. The [interview guide](INTERVIEW_GUIDE.md) explicitly separates project results, upstream functionality and AI assistance.

## Defensible portfolio claims

- Built and validated a supervised evaluation workflow for Llama 3.3 70B across NVIDIA CUDA and AMD ROCm using an existing inference runtime.
- Produced controlled four-configuration baseline measurements with independent process launches, verified placement and preserved resource/failure evidence.
- Repaired trace loss by separating complete routing spans from exact aggregate event statistics; calibrated observer overhead and checked fixed-input numerical agreement across original and instrumented runtimes.
- Used source-level and trace evidence to reject unsupported optimization explanations and preserve a profiler feasibility NO-GO.

The placement pilot later passed capacity screening, but the strict E/P numerical comparison failed and the exploratory quality retry stopped on an AMD temperature limit. No automated placement policy or original inference speedup is complete. Avoid claims of a new inference engine, custom optimized CUDA/HIP kernels, universal cross-vendor correctness, kernel-only timing, measured PCIe bandwidth or a globally optimal layer split. See [the decision](OPTIMIZATION_DECISION.md) and [portable evidence](../evidence/portfolio/) for the supported scope.
