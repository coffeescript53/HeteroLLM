> **Historical C4 proposal, preserved for provenance.** Current status through November 3: the P screening pilot completed; the frozen E/P numerical test FAILED; exploratory quality remains incomplete after a thermal stop; the performance study is BLOCKED. This document grants no execution authority and its proposed next steps are superseded by [STATUS_REPORT.md](STATUS_REPORT.md).

# C5 proposal: one bounded performance-aware placement hypothesis

**Status: CPU-only proposal; no optimizer implementation or GPU execution authorized.** C0–C4 baseline and measurement engineering is complete for review. C5 remains the targeted optimization milestone. **No specific source-level optimization is yet proven or ready to implement.** That does not mean there are no worthwhile opportunities.

## Mechanism and one candidate

The pinned [loader](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/src/llama-model.cpp#L1558-L1616) defaults to reported free-VRAM proportions, normalizes cumulative boundaries and assigns contiguous **layer-group ordinals** using `upper_bound`. It does not minimize latency, equalize measured compute, or account exactly for unequal layer bytes. The input remains on CPU; the output is a distinct group. Explicit `--tensor-split` replaces the proportions, not the kernels.

Accepted C1 E exceeded D means by 1.28% TG and 1.74% PP while moving one block toward CUDA. That small observational difference motivates a direction; it does not establish a transferable per-layer effect. C3 places a longer ROCm stream region on the serial TG path, but does not separate kernel work from waits/gaps. **Balancing device durations is not the objective:** for a serial dependency chain, a move helps only if its end-to-end marginal savings exceed its added work and overhead.

Propose **one candidate P, `-ts 3/2`**, using the original uninstrumented runtime. It moves blocks 41–48 from ROCm to CUDA: CUDA 0–48, ROCm 49–79 plus output 80, with no CPU transformer blocks. E remains CUDA 0–40, ROCm 41–79 plus output. This is the first CUDA-heavier fixed ratio already present in the old capacity-screened candidate list. The eight-block move is a bounded design choice (10% of transformer blocks), not a uniform-cost prediction, a learned optimum, or a ratio sweep. No `2/1` follow-up is proposed.

## Offline resource feasibility

Actual saved GGUF group sizes are nonuniform. At conservative context 512, F16 KV is 4,096 bytes per block/token. The 81-group float32 boundary `3/(3+2)` gives 49 CUDA groups unambiguously: `48/81 < 0.6 < 49/81`.

| Candidate P accounting | CUDA | ROCm |
|---|---:|---:|
| Encoded weights, bytes | 25,056,509,952 | 16,865,017,856 |
| Conservative KV, bytes | 102,760,448 | 65,011,712 |
| Weights + KV, GiB | 23.431396 | 15.767319 |
| Unchanged reserve, GiB | 3 | 3 |
| Minimum free VRAM for existing screen, GiB | **26.431396** | **18.767319** |
| Margin after that reserve in last accepted U snapshot, GiB | **3.972900** | **13.027603** |

The lowest CUDA free memory among the twelve accepted overhead admissions still leaves 3.942627 GiB beyond weights/KV and the 3 GiB reserve. These are archived snapshots, **not fresh admission**. Actual candidate compute/graph/staging buffers are unknown; E's compute-buffer measurements must not be copied as P's measured allocations. Retain full live checks and refuse insufficient capacity; never reduce layers or reserve to force a pass.

Host-resident encoded tensors stay 591,003,904 bytes; this is not total RSS. Keep 40 GiB available-RAM admission, 32 GiB child RSS/HWM, 4 GiB RAM floor and 256 MiB swap growth. No new host-memory estimate overrides those limits. The model remains the same 42,520,398,816-byte GGUF with SHA-256 `32df3baccb556f9840059b2528b2dee4d3d516b24afdfb9d0c56ff5f63e3a664`.

## The single proposed pilot

**One model process, candidate P only**, original U runtime, three internal repetitions of separate PP512 and depth-zero TG128. Use C0-style verbose layer/scheduler evidence to establish the new placement. All timings are **screening-only**, not formal performance observations. This one process has no contemporaneous E control; any apparent gain may include logging, time/cache/thermal or other confounding.

Exact proposed child argv (neutral paths below; private review contract binds the original executable, dependencies, model, source and output path):

```bash
"$BENCH" -m "$MODEL" -dev CUDA0/ROCm0 -sm layer -ngl 999 \
  -fitt 0 -p 512 -n 128 -d 0 -b 512 -ub 128 -t 16 \
  -ctk f16 -ctv f16 -fa off -nkvo 0 -nopo 0 --no-host 0 \
  --repack 1 -lm none -lzm off -r 3 -o json --offline -ts 3/2 -v \
  -fitc 0 -embd 0 -ncmoe 0 -mg 0 --prio 0 --delay 0 --poll 50 \
  --cpu-strict 0 -C 0x0
```

Set `GGML_SCHED_DEBUG=2` only in this supervised diagnostic child's recorded environment; neither v2 CPU tracing nor v3 timing is enabled. Preserve the original U dependency search path. Do not invoke the raw command directly. Output must be fresh `runs/llama33-70b/stage-c5-placement-pilot-20261101/01-mixed-3-2`; the proposal does not create it.

Retain a 900-second attempt including preflight and a 1,500-second total including cooldown. Existing mixed runs suggest planning several minutes of work plus bounded cooling; this is not a runtime guarantee. Hard maximum is 25 minutes. One process, no retries, replacement, alternate ratio or automatic follow-up.

Fresh source/model/runtime/dependency/hardware identities, physical GPU sensors, capacity, CPU/DIMM policy, GPU ≤45°C admission for ten seconds/eleven fresh readings after ≥30 seconds (≤600-second gate), 85°C GPU runtime stop, memory/swap limits, evidence/log caps and owned-group cleanup all remain unchanged. New write I/O errors or corrupt evidence fail immediately. Preserve all raw output and failures.

**Preparation gap, explicitly bounded:** the existing C0 validator assumes one repetition and C1 allocation validation is tied to old C0 arm profiles. Before execution approval, a small CPU-tested one-process wrapper must reuse `supervisor.py`, current cooldown and C0 placement checks, require exactly two workload rows (PP512 and TG128), each with three valid original timing samples, and enforce P's exact 49/31-plus-output map. It must not reuse E's allocation profile as if P were E, widen its tolerance, or alter frozen adapters. Bind that wrapper and its new per-placement acceptance contract in a new executable manifest. The current C5 contract is source-bound **review-only**, not an executable manifest or invented launcher command.

## Predeclared pilot decisions

- Any identity, capacity, sensor, safety, workload, placement, evidence or cleanup failure: **STOP**, preserve failure; no retry or retune.
- Passing fit/placement alone is not a performance win. To justify proceeding beyond screening, require TG mean ≥**19.196368 tokens/s** (5% above archived C1 E), PP mean ≥**703.500876 tokens/s** (no worse than 2% below E), and within-process sample CV ≤2% for both workloads. These are engineering triage gates, not confidence bounds; internal repetitions remain correlated.
- A failed threshold means **stop this candidate**. Excess variability or incomparable logging/control evidence means **inconclusive, stop for review**. A pass means only “worth separately reviewed validation,” never a causal speedup or completed C5 result. No expanded search follows automatically.

The numerical gates deliberately avoid spending further effort on a tiny or unclear historical difference. Verbose versus quiet logging can also reject a useful candidate; this conservative pilot cannot settle the performance question. A trustworthy improvement requires the later contemporaneous quiet comparison below.

## What would become original automatic policy work

Manually changing `-ts` is upstream parameter tuning. Existing `planner.py` already enumerates fixed ratios, and the older tuning path ranks measured candidates; renaming those mechanisms would not be a new contribution. Public prior art includes byte-aware scheduling and empirical placement search. C5 does not claim algorithmic novelty.

After a promising pilot and independent review, implement **one small evidence-gated boundary proposer/selector** in isolated project source, not llama.cpp kernels. Inputs: exact tensor groups/KV, current per-device capacity/reserves, source/workload-matched accepted measurements and an explicit PP regression budget. Output: one bounded CUDA-ward boundary (maximum eight blocks for this validated envelope), or abstain/retain E. Encode the chosen boundary under the pinned float32 ordinal rule; do not hardcode device names into a purported universal latency model or divide stream time by layer count. Select P only from its measured end-to-end evidence; phase timing supplies a hypothesis, not a cost oracle. CPU tests must cover nonuniform/output groups, rounding, capacity refusal, stale evidence, PP regressions, uncertainty and abstention. The differentiator is the bounded automated decision and validated operating envelope, not the existing CLI flag.

Freeze that policy and candidate before confirmation. Then, under separate approvals:

1. Compare E and P with the **same original runtime**, predeclared teacher-forced fixture, all 129 full-vocabulary float32 vectors, finite/size/hash checks, `abs(P−E) ≤ 1e−4 + 1e−5|E|`, zero violations and exact greedy IDs/token-piece bytes. Moving blocks between vendors may change floating-point arithmetic; failure stops the candidate, with no post hoc tolerance relaxation. Prior same-placement U/T equality does not establish E/P equality.
2. Run six fresh quiet counterbalanced E/P process pairs with the same workload/r3 controls and independently validated placement profiles. Use process mean elapsed times; do not reuse pilot or historical C1 as controls. Primary TG criterion: geometric P/E elapsed ratio ≤1/1.05 and nominal 95% upper bound <1. PP guard: nominal 95% upper bound ≤1/0.98. Retain all samples and order strata; these are small-sample claims for this envelope, not universal guarantees.
3. If correctness or either performance gate fails, retain E and close this candidate with a negative/inconclusive result. No ratio sweep, new profiler, kernel rewrite or months-long infrastructure extension is implicit.

C5 is complete only after an actual automatic policy is implemented, its emitted placement is validated, and fresh correctness/performance evidence is independently accepted. **Current next step: review this proposal; separately authorize the minimal CPU adapter preparation, then the exact reviewed one-process GPU pilot.** No such work is authorized by C4.
