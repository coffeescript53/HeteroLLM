# Measurement methodology

This page describes the completed 70B study and its interpretation limits. Original run artifacts and earlier proposals remain preserved. C4 is documentation and offline analysis only; no new GPU workload is authorized.

## Fixed study boundary

The study used dense Llama 3.3 70B Instruct Q4_K_M, a Ryzen 9 9950X with 96 GB installed DDR5, an RTX 5090 and a Radeon AI PRO R9700, nominally 32 GB VRAM each. The mixed CUDA/HIP llama.cpp runtime was pinned to `5e03bdd8700948b9c41c54dd1b00f28a2aebc03f`. Both single-GPU baselines used this same mixed binary and CPU backend, reducing build differences between arms.

The verified GGUF is 42,520,398,816 bytes with SHA-256 `32df3baccb556f9840059b2528b2dee4d3d516b24afdfb9d0c56ff5f63e3a664`. No result substitutes another model, quantization or historical 192 GB/48 GB configuration. The 96 GB memory configuration was not formally stability-certified. Accepted captures and tests establish their stated scope, not a hardware reliability certification.

## C1: controlled baseline measurements

| ID | Configuration | Requested GPU layers |
| --- | --- | ---: |
| A | CUDA0 + CPU/host RAM | 56 |
| B | ROCm0 + CPU/host RAM | 59 |
| D | Default heterogeneous CUDA0/ROCm0 layer split | 999, all eligible layers |
| E | Explicit equal `1/1` CUDA0/ROCm0 layer split | 999, all eligible layers |

The mixed arms differed by one transformer block in the C0 placement reference: D used CUDA blocks 0–39 and ROCm blocks 40–79 plus output; E used CUDA blocks 0–40 and ROCm blocks 41–79 plus output. Quiet C1 allocations corroborated that reference. Equal requested proportions do not require equal transformer-layer counts: the loader's partition includes the output group. Runtime placement evidence, rather than the policy name, determines actual ownership.

| Control | Value |
| --- | --- |
| Prompt processing | PP512 synthetic-token workload |
| Token generation | TG128 with initial depth zero; independent of PP512 prompt |
| Batch / microbatch / CPU threads | 512 / 128 / 16 |
| K/V cache | F16/F16, ordinary offload enabled |
| Flash Attention | Off |
| Loading | `-lm none -lzm off`, no mmap population |
| Fitting | Disabled; fixed single-GPU suffix counts |
| Warmup | Existing upstream full PP warmup and one-token TG warmup |
| Measurement | Three internal repetitions per workload in each new process |
| Independent observations | Three processes per configuration; twelve total |
| Process order | `E D A B → D B E A → B A D E` |

PP and TG are separate benchmark cases, not a 512-token prompt followed by 128 decoded tokens. They do not measure a coherent conversation, time to first token or end-to-end service latency. Loading and warmup are excluded from the original benchmark timers but included in process deadlines and resource observations. Verbose scheduler logging was disabled during C1 timing.

For each repetition, throughput is `tokens × 10^9 / elapsed_ns`. A process observation is the arithmetic mean of its three repetition throughputs. Report all observations and the mean/sample standard deviation across the three process means. These are **three independent launches**, not nine independent experiments. Mean throughput is not interchangeable with `tokens / mean(elapsed)`.

C0 established detailed placement and allocations before C1. Quiet C1 launches then checked runtime controls, capacity and independently sampled allocation/activity against that reference; they did not capture a new verbose per-layer scheduler trace in every timed process. Original C0 diagnostic timings and earlier failed/partial C1 attempts are excluded from the formal comparison. The order reduces some directional drift, but three rounds cannot form a complete four-treatment Latin square. With this small sample, observed differences do not establish a general optimization improvement.

## C2: recorder validation and observer effect

The optional recorder v2 preserves required detailed routing, copy, graph and synchronization spans while aggregating frequent low-level event bookkeeping exactly by backend. Strict validation rejects unexpected drops, overflow, incomplete scopes, malformed metadata, inconsistent aggregate counts/histograms and write failures. Intentional aggregation is distinguished from lost records. The failed v1 capacity-exhaustion capture remains a failed capture; absent retained copy spans were never interpreted as zero transfers.

Recorder v3 adds explicit warmup/PP512/TG128 boundaries and completed native CUDA/HIP event pairs. It adds no synchronization wait solely to obtain a successful capture. Every required graph/copy timing record must match the CPU trace and complete in its declared phase. Native elapsed time is measured within one device clock domain. Independent vendor device timestamps are never subtracted.

Each device interval is enclosed between the CPU time before its start-marker submission and the CPU observation of completed events. That enclosure is correlation uncertainty, not precise clock alignment. Stream elapsed intervals can contain dependency waits, contention, queue gaps or submission effects; they are not kernel-only time. Overlapping CPU enclosures do not demonstrate cross-vendor execution overlap. Logical tensor bytes are not measured physical PCIe traffic. Nested CPU scopes must not be summed as independent wall time.

### Numerical comparison

Original runtime U and instrumented runtime T processed the same fixed teacher-forced inputs in two supervised processes. The comparison covered one PP final-position vector and 128 depth-zero TG positions, each with 128,256 float32 logits: 16,545,024 values. Acceptance was predeclared as `abs(T-U) ≤ 1e-4 + 1e-5 × abs(U)` at every value, exact greedy token IDs and exact decoded token-piece bytes at all 129 positions. Sizes, positions, finite values, metadata, hashes and trace completeness were validated separately.

The accepted captures were byte-identical. This is strong evidence for that fixed input/protocol; it does not establish general numerical equivalence, free-running generation equivalence or universal model correctness.

### Paired overhead calibration

Six contemporaneous U/T blocks used counterbalanced order `UT/TU/UT/TU/UT/TU`: twelve processes with three PP and three TG repetitions each. Historical C1 throughput was not substituted for U. The process-level phase elapsed statistic is the mean of its three integer durations. For block `i`, the ratio is `T_i/U_i`; the primary estimate is `100 × (exp(mean(log(T_i/U_i))) - 1)`.

The nominal two-sided 95% interval uses Student's t with five degrees of freedom on the six log ratios, then transforms back to percent. All pairs and UT/TU strata remain visible. Accepted overhead was PP512 **+0.108100%** (−0.496609% to +0.716484%) and TG128 **+0.605006%** (+0.239675% to +0.971668%). Six blocks, possible serial dependence, order effects and two endpoints limit the interval's interpretation. The result calibrates this observer on this workload; it does not identify a causal bottleneck or demonstrate an optimization speedup.

Loading, cache state, short TG warmup, thermal drift and background activity remain confounders. End-to-end process time is recorded separately; it cannot be decomposed into isolated loading and trace-flush time where those quantities were not measured. No cache purge, clock change or power-setting intervention was used to manufacture a comparison.

## Safety and evidence boundaries

The evolved supervisor binds source, model, executable/library identities, physical GPUs, workload and hardware controls. A prior complete model checksum is reused with stat identity during launches; this avoids repeated 42.52 GB reads but does not defend against adversarial metadata-preserving replacement. Hardware identity permits only the reviewed ±1 MiB MemTotal accounting drift; other pinned fields remain exact.

Current safeguards include 40 GiB available-RAM admission, 32 GiB child RSS/HWM ceiling, 4 GiB available-RAM floor, 256 MiB cumulative swap-growth limit, GPU capacity reserves, mandatory sensor readings, a runtime GPU stop at 85°C, bounded deadlines, evidence limits and owned-process-group cleanup. GPU admission waits at least 30 seconds and requires both cards at or below 45°C for at least ten seconds across eleven fresh samples, bounded by 600 seconds and the suite deadline. Missing or invalid telemetry is a failure, never a fabricated zero.

The CPU/DIMM guard was added after C1; do not claim that C1 had temperature records that it did not collect. Later C2 captures include CPU and both DIMM SPD-hub histories. CPU protection uses the 95°C Ryzen limit. DIMM safeguards warn at 55°C, stop after 25 seconds at/above 60°C, and stop immediately at/above 65°C or a critical alarm, with 2°C hysteresis and independent timers. These DIMM values are provisional safeguards, not manufacturer-certified limits. A potentially sticky high-temperature warning alarm alone is not a critical failure; sensor settings are not changed.

RSS/HWM, system memory and whole-device VRAM are different measurements. Sampled telemetry can miss brief peaks, and device memory counters include unrelated allocations. Durable supervisor records preserve stop intent, errors, tracebacks and cleanup status. Buffered tracing can lose spans on abrupt termination; a missing/incomplete footer invalidates the capture even after otherwise successful model work.

## C3 interpretation and stopping decision

The accepted traces narrow candidate explanations but do not isolate kernel time, exact cross-vendor overlap, memory-bandwidth utilization or the counterfactual cost of removing a transfer. Long synchronization calls can be waiting for useful queued work. A long ROCm graph-stream interval is a useful localization, not proof of a particular slow kernel or a safely removable delay.

C3.1 examined existing upstream mechanisms and prior art. C3.2 ended with a source-level **NO-GO** for the proposed ROCprofv3 route under the bounded completeness/memory contract. Its CPU counterexamples establish the reviewed source concerns, not an installed-GPU profiler reproduction. No new profiler, ratio sweep or speculative transfer optimization follows from these findings. C4 completes the supported measurement/safety portfolio. The [C5 proposal](C5_PLACEMENT_PROPOSAL.md) preserves the optimization objective through one capacity-screened placement hypothesis, with separate execution approval and correctness/paired confirmation required.
