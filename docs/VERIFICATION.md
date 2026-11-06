# Verification record

Portfolio preparation: 4 November 2026. No GPU experiment, model load, driver change, or new throughput measurement was performed.

## Preserved workstation result

The uploaded license-audit logs record **205 passing CPU tests and one expected private-archive skip**: 181 passes in the selected host suite, 17 archived C3 analysis checks, and seven standalone C++ recorder checks. This is a result from the original workstation environment. The unchanged source, fixtures, recorded measurements, and original license notices are preserved.

## Fresh cloud-container checks

| Check | Observed result |
|---|---|
| Trace and portfolio-data subset | 15 passed; one expected private-archive skip |
| Archived C3 synthetic trace analysis | 17 passed |
| Standalone C++17 recorder fixtures | Seven passed |
| Full selected host suite | 182 methods: 153 passed, 27 failed methods, one error, one skip; not an all-pass run |

The trace/data subset overlaps the selected host suite and must not be added to it as unique tests. Its 27 failed methods generated 32 failure events because some methods contain subtests.

The container does not expose `/proc/pressure/memory`. The production supervisor requires that real Linux PSI telemetry and refuses to release its gated child without it. As a result, host-dependent cases stop at a monitoring failure instead of reaching their intended scenario. One descendant-cleanup case errors because the child never emits its expected output. No telemetry was fabricated and no production safety check was disabled.

The public host test script now reports that missing prerequisite before running the suite. `scripts/test-offline.sh` provides the separate analysis-only path. The original test assertions are unchanged.

## Portfolio-specific checks

`scripts/demo.py` recomputes C1 process-level means and sample standard deviations, then C2 geometric elapsed overhead and paired intervals from the included integer timings. Each result is checked against the archived summary. This replays arithmetic; the original raw full-vocabulary logit arrays and raw trace archives are not included.

The static page uses local assets and copied evidence. The build includes a source ZIP with the project's code, tests, patches, documentation, and notices. GitHub CI is configured but cannot be described as passing on GitHub until it has actually run there. The Pages workflow requires a manual trigger.

The final source manifest records file hashes for this revised presentation package. The original licensed-source manifest and delivery note are retained under `docs/release_history/`, so the old audit is not presented as an audit of new presentation files.
