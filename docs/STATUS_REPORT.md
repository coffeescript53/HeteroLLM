# Project status

Public-release preparation: 4 November 2026. Research checkpoint: 3 November 2026. No new GPU experiment was performed for this release.

| Milestone | Status and supported claim |
|---|---|
| C0–C1: mixed-vendor execution and baselines | Complete. Llama 3.3 70B ran across CUDA and ROCm; twelve independent launches compare four fixed configurations. |
| C2: measurement engineering | Complete. Recorder repair eliminated unexpected loss in the accepted follow-up; fixed-input original/instrumented captures were byte-identical; six paired blocks calibrated observer overhead. |
| C3–C3.2: attribution and profiler feasibility | Complete within scope. A longer ROCm stream region was located, but no causal removable cost was established. The proposed ROCprofv3 route ended in NO-GO. |
| C4: reproducible portfolio | Baseline source, charts, architecture, CPU reproduction and contribution boundaries are curated for publication review. |
| C5.2: placement screening | Positive capacity/functional screen for the CUDA-heavier P placement. Historical timing differences are not causal speedups. |
| C5.4: strict E/P numerical comparison | **FAILED** under the frozen criterion. The result remains unchanged. |
| Exploratory quality experiment | **Incomplete.** R2 stopped before model execution on DIMM sensor identity. After the bounded identity correction, R3 loaded E and partially ran, then stopped at 86°C AMD junction against the unchanged 85°C cutoff. P did not launch; cleanup completed. |
| Controlled E/P performance comparison | **BLOCKED.** No completed optimization, quality-equivalence finding or fresh E/P performance result exists. |

The latest R3 temperature-limit event has an unresolved cause. The fan was running at 1,214 RPM, sampled 0.096 seconds before the peak temperature; no stall was recorded. Those asynchronously read observations do not establish cooling adequacy or a manufacturer-limit violation. Historical results and failed attempts remain preserved.

The demonstrable portfolio contribution is measurement and reliability engineering around an existing runtime. A substantive inference optimization remains a separate, unfinished objective. See [results](RESULTS.md), [contributions](CONTRIBUTIONS.md) and [decision](OPTIMIZATION_DECISION.md).

The collaboration is now owner + Codex. Historical independent artifact-review findings retain their attribution; new Codex checks are not independent review. GPU work requires specific fresh owner authorization. Publication requires explicit approval and resolution of the [release checklist](PUBLIC_RELEASE.md); this package is not a published repository.
