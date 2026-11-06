# Optional CUDA/HIP measurement instrumentation

Validated measurement status (unchanged through 3 November 2026): the isolated v2 routing recorder and v3 phase/completed-device-timing recorder have run successfully on the fixed 70B equal-placement diagnostic. Independent review accepted complete CPU routing and same-device stream timing within the stated scope. A separate fixed-input U/T comparison found byte-identical full logits and predictions; six paired U/T blocks calibrated the enabled v3 observer effect. The frozen original runtime, failed v1 capture and all later runtimes remain unchanged.

| Version / files | Purpose and status |
| --- | --- |
| Root header/patch | Historical v1 detailed CPU recorder. Its GPU capture overflowed and was rejected. Preserve it for failure analysis; do not use it as the validated recorder. |
| `v2/heterollm-cpu-trace.h`, `v2/ggml-cpu-trace.patch` | Strict CPU routing/copy spans with exact bounded aggregate statistics for repetitive event bookkeeping. Complete real capture accepted. |
| `v3/ht3-*.h`, `v3/phase-timing.patch` | Full combined patch, including v2, plus workload-phase markers and bounded completed CUDA/HIP event timing. Complete capture accepted. |
| `calibration/equivalence.cpp`, `calibration/tokens.txt` | Deterministic teacher-forced comparison harness and public input-token fixture. Input fixture contains no authentication token. |

The original project contribution is these observational hooks, bounded recorder formats, strict matching/completeness checks and their validation. Model loading, quantized kernels, layer allocation and graph execution remain upstream llama.cpp/GGML. No new transfer algorithm or performance optimization is claimed. Upstream patch contexts retain the [MIT notice](../docs/UPSTREAM_LICENSE.txt); original HeteroLLM additions use [MIT](../LICENSE). Both notices must accompany redistributed patches or patched source; upstream context is not claimed as original work.

## Validated semantics and limits

The v2 recorder intentionally aggregates `event.record_submit` and `event.synchronize` by backend: exact count, total CPU-call duration, maximum and bounded histogram. Required copy/routing/graph spans remain detailed. Aggregate totals are CPU-call durations, not GPU execution time. Overflow, incomplete spans, malformed metadata, missing footer, inconsistent counts/histograms or write errors invalidate the trace. A complete zero-host-staging trace would be a legitimate negative result.

V3 adds explicit warmup/PP/TG phase attribution, native same-device event-pair elapsed time and CPU submission/completion enclosures. It preserves original calls, dependencies and waits. Events are queried at existing completion boundaries; an unresolved capture fails rather than inserting a new synchronization to force success. Full v2/v3 acceptance requires matching graph submissions and host-stage copy legs. Independent vendor clock timestamps are never subtracted.

Completed graph-stream intervals include possible waits, gaps and contention. They are not kernel-only busy time. Correlation enclosures do not demonstrate precise cross-vendor overlap. Logical tensor bytes are not measured PCIe traffic. Parent/child CPU spans overlap and must not be summed as independent costs. Whole-process event aggregates cannot be retroactively assigned to phases.

Fixed-input correctness covered 129 complete vectors, 16,545,024 float32 logits, exact greedy predictions and token-piece bytes. It is not general numerical equivalence. Accepted enabled-v3 overhead was PP512 +0.108100% (nominal 95% interval −0.496609% to +0.716484%) and TG128 +0.605006% (+0.239675% to +0.971668%), from six counterbalanced independent process pairs. This calibration does not establish a causal bottleneck or speedup. See [methodology](../docs/METHODOLOGY.md) and [results](../docs/RESULTS.md).

Builds require separate pinned checkouts, exact dependency identities and the existing host thermal safeguards. GPU execution additionally requires a fresh reviewed source-bound contract and explicit approval. [REPRODUCE.md](../docs/REPRODUCE.md) distinguishes portable CPU tests from upstream-header/library-dependent fake-backend tests. These files do not grant execution authorization.

## Historical v1 design record

The remainder describes the original v1 candidate **as it existed before GPU validation**. Its old candidate status, unknown-phase limitation and 65,536 detailed-record strategy belong to that version, not the accepted v2/v3 recorder. The failed v1 outcome is preserved; the text below is retained to make the engineering transition auditable.

This directory contains a reviewable patch against llama.cpp revision
`5e03bdd8700948b9c41c54dd1b00f28a2aebc03f`. The pinned checkout and existing
CUDA/ROCm binaries have not been modified. The patch is observational candidate
code that, at initial preparation, had only been CPU-tested and syntax-checked.
Its subsequent v1 GPU capture overflowed and was rejected; the accepted recorder
results apply to v2/v3 as described above, not to this historical v1 design.

`build_patch.py` requires exact SHA-256
`cdb79f2937aa147084a76f949adc0eb760ad1883e0ccc015bb4e0f599c89bd5f` for the
original `ggml/src/ggml-backend.cpp`. It produces `ggml-cpu-trace.patch`, which
adds the standalone header and hooks. Tests apply the patch only to a temporary
copy. Any future build must use an isolated checkout and a newly verified
source/build manifest; never replace the frozen C1 runtime in place.

## Activation and scope

The default compile has no tracing: macros expand to `((void)0)` and do not
evaluate their arguments. An isolated diagnostic build must compile the patched
GGML base translation unit with `-DHETEROLLM_CPU_TRACE=1`. At runtime an absolute
`HETEROLLM_TRACE_FILE` selects a **fresh** output file; absent/empty selects no
recording. Enabled-build/no-environment mode still incurs argument evaluation
and singleton checks, so it is not the zero-overhead control.

Do not copy this header into multiple runtime DSOs or the benchmark executable:
it intentionally owns one recorder in the GGML base translation unit. Separate
recorders could collide on the output path. Workload/phase markers need a
reviewed exported recorder API or an independently correlated marker stream,
not another header singleton. No inference command is executed by these files.

## Hook meanings (line references are the unpatched pinned source)

| Event | Hook | What the CPU span establishes |
|---|---|---|
| `backend.synchronize` | backend synchronization, 565–571 | Host time inside an existing synchronization; not an added wait. |
| `event.synchronize` | 696–700 | Host wait for an existing event. |
| `event.wait_submit`, `event.record_submit` | 689–707 | Host cost of queuing an event operation; not device wait duration. |
| `graph.submit` | 601–604 | Host graph-compute call, which can include backend-internal work; not completed GPU compute time. |
| `copy.scheduler_async_attempt` | actual scheduler copy branch, 1941 | Destination interface attempt with source/destination names and logical tensor bytes. |
| `copy.scheduler_async_accepted` | successful branch of that attempt | Marker that the interface returned true; not proof of zero-copy, peer transport, or GPU completion. |
| `copy.scheduler_fallback` | 1942–1948 | Host scope covering existing source/destination/event waits plus blocking copy. |
| `copy.public_async_fallback` | public helper, 666–669 | Separate public-helper fallback; do not confuse it with the scheduler's direct interface path. |
| `copy.buffer_interface_attempt/accepted` | 356–360 | Buffer-interface copy attempt and returned-true marker; transport mechanism remains backend-specific. |
| `copy.host_to_backend`, `copy.backend_to_host` | 635–638 | Blocking host-buffer set/get path and logical bytes. |
| `copy.host_stage` | 639–647 | Confirmed malloc/get/set/free host-staging path. |
| `copy.host_stage.get/set` | 644–645 | Nested host elapsed times for the two staging legs. |
| `graph.split_build`, `graph.allocate`, `graph.reallocate` | 1206,1747,1767 | Host scheduling/allocation scope; reallocation reason is backend change or allocation retry. |
| `graph.compute_splits`, `split.cpu_scope` | 1802–2001 | CPU call envelope and split context; GPU work may outlive them. |

The scheduler predicate is evaluated once with the original short-circuit
semantics. Return values, branches, buffer lifetimes, backend calls and existing
synchronizations are preserved. There are no added CUDA/HIP events or calls,
new synchronization, changes to placement/graph allocation, or safety-policy
edits. The enabled recorder introduces CPU work, locking and memory; those can
perturb execution and must be measured before interpreting any throughput.

## JSONL contract

Header: `type=header`, `schema=heterollm.cpu_trace.v1`,
`clock=steady_clock_ns`, `time_domain=cpu`, `phase=unknown`, capacity and
record size. A maximum of 65,536 records is preallocated (176 bytes per record
on the tested compiler: 11 MiB). Test builds may lower the capacity; increasing
it beyond 65,536 is rejected at compile time.

Each span has `seq`, hashed CPU `thread`, `kind`, `begin_ns`, `end_ns`, `graph`,
`split`, `copy`, `n_copies`, `src`, `dst`, and `logical_bytes`. `seq` is the
serialized reservation order, with monotonic begin timestamps. `graph` is an
invocation counter for compute-splits, not a persistent graph identity; split
IDs are local to that invocation. Build/allocation events outside compute-splits
have graph zero. `copy`/`n_copies` expose pipeline-buffer selection, not proof of
overlap. Buffer-path names are buffer names; scheduler names are backend names.
Thread hashes are process-local correlation aids, not OS TIDs.

Nested scopes overlap. **Do not sum parent and child durations or bytes.** Count
directional logical requests using scheduler accepted/fallback markers. Count
host staging using `copy.host_stage`; its two legs carry the same payload bytes
each. Physical PCIe traffic, transfer-engine occupancy, and bytes moved by other
internal backend APIs remain unmeasured. Interface-attempt plus accepted marker
are one attempted operation, not two transfers. No data copied is sampled.

The footer reports `recorded`, `dropped`, `unfinished`, `metadata_truncated` and
`complete`. Overflow, unfinished spans or truncated names make `complete=false`
and emit `HETEROLLM_TRACE_ERROR` on stderr. Missing header/footer, malformed
records, inconsistent counts, clock ordering, any error marker, or a failed
supervisor run must invalidate the diagnostic. Overflow never silently rolls
over or changes inference behavior. The reviewer/adapter must reject an
incomplete trace; this patch alone is not a supervisor adapter.

## Durability, overhead and known limits

The output uses `O_EXCL` and mode 0600 to prevent overwriting earlier evidence.
The header is flushed/fsynced at initialization. Span data stays in bounded
memory and is serialized/flushed/fsynced at orderly process exit. A crash,
SIGTERM/SIGKILL or `_exit` may leave **only the header**; absence of a footer is
an explicit unusable-trace condition. Existing external supervisor telemetry,
stop intent, traceback, exit/cleanup records remain the durable failure record.
No signal handlers, timers, child processes or system settings are added.

Future GPU proposals must retain evidence-size caps and account for the bounded
final flush. Filesystem failure can occur after a `complete:true` footer was
buffered; therefore stderr/readback/supervisor status are required in addition
to the footer. No assertion about durable completion follows from JSON alone.

This candidate cannot establish GPU stage completion, compute/transfer overlap,
GPU idle fractions, or prefill-versus-decode cost. All phases (initial build,
warmup, prefill, generation, depth prefill) are explicitly **unknown**. Neither
node counts nor durations may be used to invent phase labels. Add and validate
phase markers plus CUDA/HIP completion-aware timeline capture in a separate
reviewed patch/design before claiming those quantities. In particular, a CPU
submission interval is not a GPU kernel duration, and host wait includes
previous queued work; it is not necessarily avoidable synchronization cost.

CPU probe (bounded, no model/GPU):

```bash
python3 -m unittest discover -s tests -p test_trace_candidate.py -v
python3 instrumentation/measure_cpu_cost.py --output /tmp/fresh-cpu-trace-cost.json
```

The probe runs 10,000 no-op recorder begin/end pairs once and reports
initialization, loop, and serialization/durability costs separately versus a
compile-disabled checksum loop. It is a synthetic CPU cost sample, not an
estimate of GPU inference overhead or a statistical comparison. GPU overhead
requires separately authorized, independent matched control/trace launches
under the same safety, placement, workload and thermal protocol.

Tests cover compile-disabled argument suppression/no writes, nested timing,
escaped metadata, per-thread contexts, bounded overflow, incomplete/truncated
records, output collisions, abrupt termination, exact source binding, clean
patch applicability and enabled/disabled syntax of the complete pinned
`ggml-backend.cpp`. They do not validate a newly linked CUDA/ROCm runtime.
