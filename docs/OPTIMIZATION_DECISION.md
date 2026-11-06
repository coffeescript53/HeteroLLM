# Optimization decision and remaining milestone

Research checkpoint: 3 November 2026 · Existing evidence and source only

**No specific source-level optimization is yet proven or ready to implement.** This is not a claim that HeteroLLM has no worthwhile remaining optimization opportunities. C4 completes the reproducible baseline and measurement-engineering portfolio. C5's placement pilot passed screening, but strict cross-placement numerical validation failed and exploratory quality remains incomplete. No automatic policy or causal speedup is claimed.

This conclusion concerns the inspected workload and mechanisms. The original [C5 proposal](C5_PLACEMENT_PROPOSAL.md) is historical rationale, not current execution authorization. The twelve-process E/P performance experiment remains blocked.

## Decision evidence

The portable [results and provenance](../evidence/portfolio/) preserve the accepted C1–C3 measurements used here. C1 and C2 results, C3 attribution and C3.1 prior-art findings received independent artifact review within their recorded scopes. C3.2 is a completed source/CPU NO-GO. Later Codex-reported checks and upload integrity are not independent scientific validation.

| Candidate direction | What is established | Why implementation is not justified |
|---|---|---|
| Replace cross-vendor host staging | The measured route is CUDA → host → ROCm. TG parent staging CPU coverage averages 69.060 ms in a 6,950.354 ms phase across six instrumented processes. | This inclusive CPU scope is approximately 0.993% of the phase, not physical transfer time or an achievable savings bound. No material avoidable transfer cost or safe replacement lifetime protocol was established. |
| Remove synchronization | The scheduler fallback averages 2,116.412 ms, including 2,047.175 ms of nested backend synchronization. | Much of the wait covers useful producer completion. Source and destination reuse protections are required; positive wait duration is not proof of redundancy. |
| Enable graphs or improve generic graph setup | TG graph replay is already active on both GPUs. Scoped split/build/allocate work is small in PP and unrecorded in measured TG. | A generic enable-graphs change duplicates existing behavior. Uncaptured PP calls are not proof of a defective graph cache. Newer graph chunking is known upstream work, not a new HeteroLLM mechanism. |
| Tune an AMD quantized kernel | ROCm completed graph-stream residence averages 4,773.728 ms per TG phase, longer than CUDA's 2,046.352 ms. Host ordering places these stages on the serial token path. | Stream residence is not kernel-only time. Actual replayed kernel/operator identities, internal gaps and removable work remain unresolved. Existing architecture-specific tuning creates substantial duplication risk. |
| Change the MMVQ dispatch threshold from eight to four columns | The inspected public proposal targets five-to-eight-column operations. | A one-column TG operation satisfies either threshold; TG128 is 128 sequential steps, not 128 concurrent columns. No improvement mechanism follows for this treatment. |
| Overlap more work | The scheduler has asynchronous capabilities and copy slots. | Capability is not an exposed opportunity. Layer dependencies, input reuse and per-token completion constrain this workload. Safe independent work has not been identified. |

These durations overlap and **must not be stacked or added as a wall-time budget**. No independent CUDA/HIP timestamps were subtracted. Logical tensor bytes are not measured PCIe traffic; allocation size is not measured DRAM traffic. A theoretical bandwidth calculation cannot turn the unresolved ROCm region into a measured roofline or a speedup estimate.

## Pinned source supports the limits

The frozen runtime is `5e03bdd8700948b9c41c54dd1b00f28a2aebc03f`.

- [Scheduler fallback](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-backend.cpp#L1938-L1948) synchronizes the producer and protects destination-slot reuse before copying.
- [Generic tensor copy](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-backend.cpp#L628-L649) uses host allocation/get/set/free when the backend buffer-copy callback cannot handle the route.
- [CUDA/HIP async-copy checks](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-cuda/ggml-cuda.cu#L2544-L2601) require compatible compiled backend instances and buffers. [Backend-local callback identity](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-cuda/ggml-cuda.cu#L741-L743), not different GUID text, distinguishes foreign-family buffers here.
- [Buffer get/set](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-cuda/ggml-cuda.cu#L779-L792) includes native copies and completion waits. Its CPU duration is not a kernel-only measurement.
- [MMVQ selection](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-cuda/mmvq.cu#L318-L428) and the [eight-column threshold](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-cuda/mmvq.cuh#L3) show eligibility, not the kernel actually selected in every captured graph.

The existing C3.1 audit compared public work at fixed revisions as of 2026-10-30. It is an applicability check, not a proof of universal novelty or a claim about future upstream changes. C4 performed no new broad search.

## Why the C3.2 profiler route remains closed

The chosen ROCprofv3 route did not meet the fixed loss-completeness, nonblocking and aggregate memory/output requirements. Exact-source CPU counterexamples showed an uncounted rejection path and callback/finalization limitations; typed ring capacity alone exceeded the proposed collector budget before other ownership. This was not a measured RSS excursion, installed-SDK GPU reproduction or permission to weaken the contract.

A small tool adaptation could not close SDK-level gaps. Reopening that profiler investigation or building a custom collector is outside the completion scope. The NO-GO remains intact. A placement pilot using the original runtime is a separate C5 hypothesis, not a restart of C3.2.

## What the placement evidence changes

The original runtime's default `--tensor-split` policy distributes by available memory; it is not a measured per-backend service-time optimizer. E requests equal weighting and places CUDA blocks 0–40, ROCm 41–79 plus output. P requests 3:2 and places CUDA blocks 0–48, ROCm 49–79 plus output. Neither split balances kernel time by construction, and the longer ROCm graph-stream region does not establish uniform per-layer costs.

P's one-process screening result was 853.196306 PP512 and 20.286076 TG128 tokens/s, approximately +18.85% and +10.96% against historical E. Those older observations are not contemporaneous controls. C5.4 then failed the frozen E-referenced numerical criterion with 16,448,640 violations and 128/129 greedy predictions matching. The later quality experiment has no valid E/P result: R3 stopped during E at an AMD junction temperature of 86°C against the 85°C cutoff; P did not run.

These outcomes do not prove that moving the boundary is inherently invalid or that a model-accuracy regression occurred. They also do not justify dismissing the discrepancy, relaxing the original test or claiming quality preservation. The original failure stays failed.

## Two finish lines

**C4:** complete the accepted baseline/measurement narrative, charts, reproducibility, public-source review and accurate contribution boundaries. No new GPU run is needed for that milestone.

**C5:** resolve whether the placement hypothesis warrants further work before selecting a differentiated source-level mechanism. Manual `-ts` tuning is not itself an original optimizer. The existing fixed-ratio planner and empirical ranking are prior project capabilities; a new mechanism must explain what it changes beyond those and upstream llama.cpp, preserve correctness and justify its added complexity with contemporaneous measurements.

No new implementation or GPU experiment is part of this public-release task. The next research action requires a separate owner decision about the unresolved thermal behavior and quality measurement; there is no automatic retry, performance run, ratio sweep or profiler restart. If future evidence is weak or the mechanism duplicates existing capabilities, STOP/PIVOT remains the defensible outcome. Publication of the completed measurement work does not declare the optimization objective complete.
