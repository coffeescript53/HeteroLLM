#!/usr/bin/env python3
"""Recompute published summaries from bundled timings; never run inference."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def check(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Recomputed value differs from published summary: {actual} vs {expected}")


def main() -> None:
    data = json.loads((ROOT / "evidence/portfolio/results.json").read_text())
    raw = json.loads((ROOT / "evidence/portfolio/raw-timings.json").read_text())
    processes = raw["datasets"]["c1"]["processes"]
    if len(processes) != 12:
        raise ValueError("Expected twelve independent baseline processes")
    print("HeteroLLM | offline evidence replay")
    print("Recomputing recorded observations; no GPU or new benchmark is used.\n")
    print(f"{'Configuration':<23} {'PP512 tokens/s':>21} {'TG128 tokens/s':>21}")
    for config in ("A", "B", "D", "E"):
        group = data["c1"]["groups"][config]
        values = []
        for workload, count in (("pp512", 512), ("tg128", 128)):
            runs = [p for p in processes if p["configuration"] == config]
            if len(runs) != 3:
                raise ValueError("Each configuration must contain three processes")
            rates = []
            for run in runs:
                durations = run["workloads"][workload]["samples_ns"]
                if len(durations) != 3 or any(type(t) is not int or t <= 0 for t in durations):
                    raise ValueError("Expected three positive integer durations per process/workload")
                rates.append(mean(count * 1e9 / t for t in durations))
            mu, sd = mean(rates), stdev(rates)
            check(mu, group[workload]["mean"])
            check(sd, group[workload]["sample_sd"])
            values.append(f"{mu:.3f} +/- {sd:.3f}")
        print(f"{group['label']:<23} {values[0]:>21} {values[1]:>21}")
    print("\nMeans and sample SDs across 3 independent processes/configuration.")
    print("PP512 and depth-zero TG128 are separate workloads.\n")
    paired = raw["datasets"]["c2_overhead"]["processes"]
    if len(paired) != 12:
        raise ValueError("Expected six pairs of original/instrumented processes")
    for workload in ("pp512", "tg128"):
        logs = []
        for block in range(1, 7):
            members = [p for p in paired if p["block"] == block]
            if len(members) != 2 or {p["arm"] for p in members} != {"U", "T"}:
                raise ValueError("Missing or duplicate member of overhead pair")
            elapsed = {p["arm"]: mean(p["workloads"][workload]["samples_ns"]) for p in members}
            logs.append(math.log(elapsed["T"] / elapsed["U"]))
        mu = mean(logs)
        radius = 2.570581835636305 * stdev(logs) / math.sqrt(6)
        estimate = 100 * math.expm1(mu)
        interval = [100 * math.expm1(mu - radius), 100 * math.expm1(mu + radius)]
        expected = data["c2"]["overhead"]["primary_workloads"][workload]
        check(estimate, expected["geometric_overhead_percent"])
        for a, e in zip(interval, expected["nominal_95pct_ci_percent"]):
            check(a, e)
        print(f"{workload.upper()} observer overhead: {estimate:.3f}% "
              f"(nominal 95% paired interval {interval[0]:.3f}% to {interval[1]:.3f}%)")
    print("\nAll recomputed summaries match the bundled results.")
    print("The 16.5M-logit comparison is an archived result; raw logit arrays are not bundled.")
    print("See docs/RESULTS.md for the experimental scope and limitations.")


if __name__ == "__main__":
    main()
