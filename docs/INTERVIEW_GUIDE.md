# HeteroLLM presentation and interview guide

Current portfolio narrative · research through 3 November 2026

The completed portfolio contribution is a reproducible mixed-vendor inference measurement system, led by a demonstrated trace-recorder repair. It combines Llama 3.3 70B execution, controlled baselines, calibrated instrumentation, fixed-input observer validation and durable safety/cleanup records. An original inference speedup is **not complete**: C5's strict cross-placement comparison failed and its exploratory quality retry is incomplete. C4 measurement engineering and C5 optimization are separate milestones.

## Thirty-second explanation

"I built and validated measurement tooling around a 70B model running across NVIDIA CUDA and AMD ROCm. A bounded recorder initially filled with repetitive event calls and lost the transfer evidence. I changed its strategy to aggregate those calls while preserving detailed routing spans; the accepted follow-up had zero unexpected losses. I then checked fixed-input numerical agreement and calibrated the observer with six paired runs: about 0.11% prompt and 0.61% generation overhead. llama.cpp provides inference and placement. My contribution is the instrumentation, validation, supervision and analysis; a new inference optimization remains unfinished."

Use first-person claims only for work you can explain and defend. This was an AI-assisted workflow with owner-controlled experiment authorization and historical separate ChatGPT artifact reviews. The current collaboration is owner + Codex; no external review gate is implied. Describe personal decisions and implementation ownership accurately; see [contributions](CONTRIBUTIONS.md).

## Five-minute walkthrough

**1. The systems problem.** The 42.52 GB GGUF exceeds either 32 GB GPU's capacity. The host has 96 GB RAM. Two GPU memories are separate allocations, not a transparent 64 GB pool. Layer splitting places dependent computation on different backends, so avoiding CPU offload does not imply perfect concurrency or linear scaling.

**2. The controlled comparison.** The study measured NVIDIA + RAM, AMD + RAM, default mixed and explicit equal mixed with the same verified model and pinned runtime. Fitting was disabled, loading used `none/off`, and PP512 and depth-zero TG128 were separate workloads. There were three independent processes per configuration and three internal repetitions per workload. The internal repetitions were not treated as independent experiments.

**3. What actually ran.** Physical-device checks, layer maps, backend allocations and scheduler evidence established computation on both backends. Equal mixed placed transformer blocks 0–40 on CUDA and 41–79 plus output on ROCm. “1:1” is a requested split control, not a promise of exactly forty blocks or equal service time. The verbose mapping came from C0; quiet C1 allocations corroborated it. Show the architecture and the [portable results](../evidence/portfolio/).

**4. Why trust the observer.** The first recorder exhausted 65,536 slots and dropped 22,336 spans. The repair retained 5,222 detailed spans and 82,650 exact aggregate calls with zero unexpected losses, including 137 host-staging copies carrying 37,781,504 logical bytes. Malformed/incomplete traces fail even if the model exits successfully. Native timing pairs are completed on each device without inserting waits for collection. A deterministic U/T test compared 129 full-vocabulary logit vectors; then six counterbalanced U/T pairs measured overhead separately from correctness.

**5. The decision.** The AMD graph stage is longer on the serial generation path, but its internal kernel/gap composition is unknown. Most of the scheduler fallback's CPU duration is nested producer synchronization; it is not all transfer overhead. Known dispatch and graph techniques already exist upstream or do not match this workload. The proposed native activity route ended in NO-GO. A CUDA-heavier placement later passed capacity screening, but strict numerical validation failed. The exploratory quality retry stopped during E on an AMD temperature limit; P never launched. No policy or causal inference speedup has been validated.

## Results worth remembering

These are C1 mean tokens/s ± sample SD across three independent launches per configuration. They are descriptive baselines, not a measured optimization effect.

| Configuration | PP512 tokens/s | TG128 tokens/s |
|---|---:|---:|
| NVIDIA + RAM | 221.377 ± 0.666 | 4.858 ± 0.007 |
| AMD + RAM | 174.012 ± 0.103 | 4.594 ± 0.005 |
| Default mixed | 705.573 ± 0.919 | 18.052 ± 0.030 |
| Equal mixed | 717.858 ± 2.809 | 18.282 ± 0.006 |

The mixed configurations avoid the substantial CPU offload present in the single-GPU controls. These are different placements under matched workload settings, not isolated GPU microbenchmarks. Default/equal mixed differ by one transformer block in the diagnostic mapping; their small descriptive difference does not establish an optimum.

The fixed-input equivalence result covers **16,545,024 float32 values, 129 greedy token IDs and 129 token-piece byte strings**. U and T captures were byte-identical. Predeclared acceptance allowed absolute error `1e-4 + 1e-5 × |U|`, with zero violations and exact token/piece agreement. Byte equality is stronger for this fixture; it does not prove general model correctness.

The six-pair overhead study measured elapsed-time overhead, not throughput improvement:

| Workload | Geometric paired overhead | Nominal 95% interval |
|---|---:|---:|
| PP512 | +0.108100% | −0.496609% to +0.716484% |
| TG128 | +0.605006% | +0.239675% to +0.971668% |

Six pairs are a small sample. The intervals assume approximately normal independent paired log ratios; serial effects and residual confounding remain. A PP interval spanning zero does not prove zero overhead. These estimates apply to v3 instrumentation, not an arbitrary profiler.

## Core concepts to explain precisely

**Prefill versus decoding.** PP512 processes a synthetic 512-token prompt. TG128 starts at depth zero and performs 128 sequential generation steps; it is not a 512-token prompt followed by 128 response tokens. Prompt matrix shapes expose different parallelism from one-token decoding. The benchmark executes inference kernels on synthetic inputs, not a semantic quality test.

**Model storage versus memory demand.** Q4_K_M uses mixed tensor types and block metadata. File size is not simply parameter count divided by two, and it is not peak VRAM or host RSS. Weights, KV cache, compute buffers, staging, driver allocations and page cache must be distinguished. GGUF directory accounting supports capacity estimates; runtime logs and telemetry validate actual allocations.

**Layer placement versus execution.** A tensor's storage location alone does not prove which backend executes every operation. Layer maps, buffer allocations, scheduler assignments and completed workload evidence answer different questions. Activity on both GPUs does not prove simultaneous execution or useful overlap.

**Cross-vendor copying.** The pinned backend tries compatible async copying first. Foreign CUDA/HIP buffer families fall back through host memory after producer completion and destination reuse checks. The source shares GUID text; backend-local callback identity is the relevant distinction here. Native CUDA events cannot simply be supplied to HIP. [Pinned copy path](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-backend.cpp#L1938-L1948).

**Timing domains.** CPU-call duration includes submission, blocking and runtime work. Completed native event intervals measure same-device stream residence, which can contain dependencies and gaps. They are not kernel-only activity. Independent CUDA and HIP timestamps cannot be subtracted to obtain a cross-device gap. CPU enqueue/completion bounds establish some ordering, with unresolved intervals left unknown.

**Nested scopes.** A scheduler fallback can enclose a producer synchronization and a staged copy. Summing all three counts the same time repeatedly. Use interval unions and explicit nesting; report GPU intervals separately rather than constructing an unsupported shared-clock wall-time budget.

**Logical bytes versus traffic.** Counting a tensor's payload once describes logical transferred data. It does not measure PCIe transactions, cache behavior or DRAM traffic. Similarly, a model buffer's allocated bytes cannot establish bandwidth utilization or remaining speedup.

## Statistics without pseudoreplication

For a sample containing `N` tokens and integer duration `t_ns`, throughput is `N × 10^9 / t_ns`. C1 summarizes sample throughputs within each process, then summarizes the three process values. The mean of reciprocal durations need not equal the reciprocal of the mean duration.

For overhead, each pair compares T and U **mean elapsed durations**, forms their log ratio, and summarizes six pair values. Exponentiating the mean log ratio yields a geometric elapsed ratio; subtracting one gives overhead. Counterbalanced UT/TU order reduces simple order effects, but does not eliminate all cache, thermal, loading or temporal confounders. Loading and warmup are distinguished from the benchmark's timed intervals; their isolated effects were not all measured.

Historical C1 controls are not substituted into the contemporaneous U/T study. Partial failed suites and C0 diagnostic timings are excluded from formal C1 comparisons. Original samples remain available; no post hoc tolerance change or convenient outlier removal supports the claims.

## Questions an inference engineer may ask

**What is original here?** The experiment controller, source-bound execution contracts, capacity/placement validation, durable resource and failure records, bounded optional instrumentation, strict analyzers, numerical capture validation, paired calibration and source-linked interpretation. llama.cpp supplies the inference implementation and kernels. The earlier placement-search scaffolding was tested offline; it was not validated as a tuned GPU optimizer.

**Why not optimize the host copy?** TG staged-copy parent CPU coverage is about 69.060 ms in a 6,950.354 ms phase. The much larger fallback includes about 2,047.175 ms of producer synchronization. Those numbers neither identify a dominant copy bottleneck nor bound the gain from redesigning dependencies. A safe async bridge would also need ownership, completion and buffer-reuse proofs.

**Why not tune the AMD kernel?** The longer ROCm region is credible, but executed kernel/operator identity and internal gaps are unresolved. Existing architecture-specific kernels and public patches make generic tuning a novelty risk. The known eight-to-four-column threshold proposal does not change a one-column decision. No mechanism and speedup were invented to complete the portfolio.

**Why did the profiler investigation stop?** The chosen SDK/tool route could not meet the predeclared loss-integrity and resource bounds through a small adaptation. CPU counterexamples exposed SDK-level behavior below the callback. That is a scoped NO-GO, not a GPU reproduction or a statement that all vendor tooling is unusable.

**How did the safety system avoid becoming its own source of bad results?** By preserving failures, reproducing parser/sensor/file-race bugs offline, using bounded readers and durable stop intent, verifying cleanup, and never changing thresholds merely to obtain a pass. Warnings, critical alarms and absent measurements are separate states. Sampling and software termination still cannot guarantee protection from every hardware or kernel failure.

**What would you do differently?** Introduce process-specific memory, durable cleanup and strict trace-capacity tests earlier; distinguish sensor power-state transitions from missing-hardware faults; make the research stop criteria explicit before investing in another profiler. Keep the workload fixed while debugging the observer.

**What remains unproved?** General numerical equivalence, precise cross-vendor overlap, kernel-only time, physical bus/DRAM traffic, a causal dominant bottleneck and an original runtime speedup. C1 lacks CPU/DIMM thermal histories because those monitors were added later. The 96 GB configuration was not formally stability-certified.

**Did the faster placement preserve correctness?** That has not been established. P measured 853.196306 PP512 and 20.286076 TG128 tokens/s in one screening process, but its comparison with historical E is not a contemporaneous speedup. The frozen numerical test failed at 16,448,640 entries, with 128/129 greedy predictions matching. Logit-tolerance failures are not a task-accuracy score, and a near-tie flip is not automatically harmless. The original failure stays failed; the separate exploratory quality experiment did not finish.

**Why did the latest run stop?** A corrected DIMM identity gate let E load and begin the workload. The AMD junction then reached 86°C against the unchanged 85°C stop. The fan was running at 1,214 RPM 0.096 seconds before that reading; no stall was recorded. The parent preserved partial and rejected evidence and reaped all owned processes. P did not launch. The thermal cause remains unresolved, and that outcome supplies no E/P quality comparison.

## Resume wording

Use only wording that matches personal responsibility; disclose AI assistance when describing the workflow.

- Repaired a bounded GPU-routing recorder that dropped 22,336 spans; preserved 5,222 detailed spans and exactly aggregated 82,650 event calls with zero losses in the accepted follow-up.
- Built an AI-assisted CUDA/ROCm evaluation workflow for Llama 3.3 70B; compared four deployment configurations across twelve independent launches with verified placement and preserved resource evidence.
- Validated byte-identical original/instrumented captures across 16.5 million float32 logits and calibrated observer overhead through six paired experiments: 0.108% PP512 and 0.605% TG128.

For a SWE-focused application, substitute: “Implemented source-bound execution contracts, mandatory sensor/resource gates and durable process-group cleanup; reproduced parser, atomic-file accounting and sensor-enumeration failures with CPU-only regressions.”

Avoid “optimized inference by 4×,” “invented mixed-vendor inference,” “zero-overhead profiler,” “proved the GPU was idle,” or “designed new CUDA kernels.” The mixed-versus-host-offload baseline difference is not a before/after runtime optimization.

For a presentation, show the architecture, C1 chart, equivalence/overhead summary, one nested-interval example and the final decision. A saved-results demonstration is sufficient; no live GPU launch is needed. Continue with [engineering decisions](ENGINEERING_JOURNEY.md), [contribution boundaries](CONTRIBUTIONS.md) and [the optimization decision](OPTIMIZATION_DECISION.md).
