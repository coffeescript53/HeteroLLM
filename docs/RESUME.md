# Resume entry

## Recommended version

**HeteroLLM — GPU Inference Profiling & Benchmarking**  
*Python, C++, Linux, CUDA, ROCm, llama.cpp · 2026*

- Built a Python benchmark and process supervisor around llama.cpp to evaluate Llama 3.3 70B across NVIDIA and AMD GPUs, comparing four configurations in 12 independent runs with thermal limits, timeouts, and process-group cleanup.
- Reworked C++ tracing to aggregate frequent events while preserving transfer records; reduced dropped spans from 22,336 to zero in the follow-up diagnostic and captured 137 host-staged cross-vendor transfers.
- Validated instrumentation against the original runtime on 16.5M fixed-input logits, obtaining byte-identical results; estimated 0.61% generation-time overhead across six counterbalanced run pairs.

Add the actual GitHub repository link to the project title once it exists. Use the public portfolio URL when you choose to make the page accessible to recruiters. No placeholder URL belongs on the submitted resume.

## If the resume needs a shorter entry

**HeteroLLM — GPU Inference Profiling & Benchmarking**  
*Python, C++, Linux, CUDA, ROCm*

- Built supervised 70B inference benchmarks across NVIDIA and AMD GPUs; compared four configurations in 12 independent runs with resource limits and process-group cleanup.
- Repaired C++ trace loss from 22,336 dropped spans to zero in a follow-up diagnostic; checked 16.5M logits for byte equality and measured 0.61% generation overhead across six paired runs.

## Evidence behind the wording

| Claim | Source |
|---|---|
| Supervision and cleanup | [supervisor.py](../heterollm/supervisor.py), [supervisor tests](../tests/test_supervisor.py) |
| Four configurations / 12 launches | [results](RESULTS.md), [raw timings](../evidence/portfolio/raw-timings.json) |
| Trace repair / 137 copies | [v2 recorder](../instrumentation/v2/heterollm-cpu-trace.h), [recorder evidence](../evidence/portfolio/recorder-summary.json) |
| Fixed-input equality / observer overhead | [C2 results](RESULTS.md), [portable data](../evidence/portfolio/results.json) |

The trace-loss comparison is the v2 diagnostic. The calibrated overhead is the later phase-aware v3 observer. The numerical check uses the same placement and fixed 129-position fixture. The throughput difference between mixed GPUs and CPU offload belongs in the case study, where its different memory/placement configuration is visible.

## Interview opening

“I wanted to understand the cost of running a 70B model across NVIDIA and AMD GPUs. llama.cpp handled inference; I built the experiment supervision and measurement around it. My first trace recorder filled with repeated event calls and lost the transfer records I needed. I changed those events to exact aggregates and kept detailed transfer spans, which gave me a complete follow-up capture. Then I checked the instrumentation itself: 16.5 million fixed-input logits matched byte for byte, and six paired runs estimated about 0.61 percent generation overhead. The useful engineering result was a measurement system I could validate. Placement optimization is still open.”

Development was AI-assisted, as documented in [CONTRIBUTIONS.md](CONTRIBUTIONS.md). Be ready to explain the recorder's storage tradeoff, process cleanup, independent sampling, and the limits of the numerical comparison.
