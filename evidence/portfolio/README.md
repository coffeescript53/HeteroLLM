# Public evidence projection

`results.json` is an allowlisted projection of frozen, independently reviewed C1/C2/C3 evidence. It contains numeric observations, descriptive statistics, selected identity hashes and source-file provenance. Personal paths, process identifiers, physical sensor identifiers and private review links are excluded. It does not replace the complete private research archive.

No benchmark was rerun to produce these files. The C4 projection and presentation require their own review; acceptance of the original experiments does not automatically establish correctness of a new chart.

`raw-timings.json` retains all **72 original C1 integer duration samples** and **72 original C2 overhead duration samples**, extracted directly from 24 original stdout files. Their expected SHA-256 values come from the existing accepted archive inventories. C1 records retain configuration, ordinal and round; C2 records retain ordinal, block and U/T arm. Each workload retains `n_prompt`, `n_gen`, `n_depth` and all three `samples_ns` values. These are two separate datasets, never pooled. Public tests recompute process throughput and paired elapsed means directly from these integer observations.

From the repository root, the archive owner can reproduce or check the projection:

```bash
python3 scripts/derive-portfolio-data.py --check
python3 -m unittest discover -s tests -p test_portfolio_data.py -v
python3 scripts/plot-portfolio.py
```

The first command requires the original `runs/llama33-70b` evidence tree and validates all eight archived analysis inputs plus 24 original stdout SHA-256 values before extraction. The public release intentionally excludes that private tree. Public arithmetic, schema, provenance, units, privacy and interpretation tests still run without it; only the original-archive equality test is skipped. Plotting requires Matplotlib; the data exporter and tests use only the Python standard library.

## Reading the figures

- **C1:** points are three independent process observations per configuration; each averages three correlated internal rates. Error bars are between-process sample SD, not confidence intervals. PP512 and TG128 have separate scales. The four placements are descriptive baselines, not an original runtime speedup.
- **C2:** small points are six paired process elapsed-time overheads. Diamonds and error bars are the geometric mean ratio and nominal 95% Student-t interval on paired log ratios. Internal repetitions are not independent observations. Six pairs, order effects and uncontrolled background/thermal trends limit generalization.
- **C3:** separate rows are inclusive measurements that overlap. They must not be stacked or summed into a wall-time budget. Error bars are between-process sample SD across six T processes. Completed same-device stream intervals include dependencies and gaps, not kernel-only execution; CPU synchronization is not wasted GPU time. The logarithmic duration axis makes small copy scopes visible without suggesting they fill the entire phase. Logical bytes are not physical PCIe traffic.

All memory values in the data are bytes; GiB means bytes divided by 2³⁰. Resource maxima are sampled except explicitly named wait4 lifetime RSS. Total device VRAM includes background allocations. Layer maps are the C0 observed reference corroborated by quiet C1 allocation/activity; indices 0–79 are transformer blocks, index 80 is the output layer.

The supported attribution is a longer ROCm graph region on the serialized TG dependency path. Its internal kernel/gap breakdown and a removable dominant bottleneck remain unknown. No new optimization implementation or GPU execution is part of C4. `c5-feasibility.json` separately records archived capacity arithmetic for one proposed C5 placement; it is not a fresh admission, measured improvement or authorization.

## Recorder repair evidence

`recorder-summary.json` contains the exact v1/v2 headers, footers and v2 aggregate groups, trace hashes, and counts of retained host-stage spans. V1 dropped22,336 records; its actual transfer count remains UNKNOWN. V2 has5,222 complete detailed spans,82,650 aggregate calls and137 host-stage spans carrying37,781,504logicalbytes. This compact derivative was checked against both original traces during release preparation; their recorded outcomes are unchanged. It is not a replayable raw trace or a new performance measurement.
