#!/usr/bin/env python3
"""Render the committed public projection; never read devices or run workloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"A": "#33658a", "B": "#aa6147", "D": "#7a7398", "E": "#208477"}
INK = "#172b3a"
MUTED = "#526777"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": INK,
        "axes.edgecolor": "#a8b6bf", "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "svg.hashsalt": "heterollm-c4",
    })


def save(fig, output, name):
    fig.savefig(output / f"{name}.svg", metadata={"Date": None}, facecolor="white")
    fig.savefig(output / f"{name}.png", dpi=180, metadata={"Software": "HeteroLLM public evidence plotter"}, facecolor="white")
    plt.close(fig)


def c1(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.7))
    fig.subplots_adjust(left=.09, right=.98, top=.76, bottom=.25, wspace=.30)
    fig.text(.07, .94, "70B baseline throughput across four placements", fontsize=17, weight="bold")
    fig.text(.07, .885, "Llama 3.3 70B Q4_K_M · RTX 5090 + Radeon R9700 · 96 GB host RAM", fontsize=10.5, color=MUTED)
    keys = ["A", "B", "D", "E"]
    labels = ["RTX 5090\n+ RAM", "R9700\n+ RAM", "Default\nmixed", "Equal\nmixed"]
    for ax, workload, title in zip(axes, ("pp512", "tg128"), ("PP512 · prompt processing", "TG128 · token generation")):
        for i, key in enumerate(keys):
            group = data["c1"]["groups"][key][workload]
            ax.bar(i, group["mean"], width=.58, color=COLORS[key], alpha=.20, zorder=2)
            ax.errorbar(i, group["mean"], yerr=group["sample_sd"], color=INK, capsize=5, lw=1.5, zorder=5)
            ax.scatter([i - .11, i, i + .11], group["independent_launch_means"], s=30, color=COLORS[key], edgecolors="white", linewidths=.4, zorder=4)
            ax.text(i, group["mean"] * 1.035 + (max(data["c1"]["groups"][k][workload]["mean"] for k in keys) * .01), f'{group["mean"]:.2f}', ha="center", fontsize=10, weight="bold")
        ax.set_xticks(range(4), labels)
        ax.set_ylim(0, max(data["c1"]["groups"][k][workload]["mean"] for k in keys) * 1.18)
        ax.set_ylabel("Tokens / second")
        ax.set_title(title, fontsize=12, loc="left", pad=14)
        ax.grid(axis="y", color="#e5eaee", zorder=0)
        ax.set_axisbelow(True)
    fig.text(.07, .14, "Points: 3 independent process observations per configuration. Bars: mean; whiskers: sample SD.", fontsize=10)
    fig.text(.07, .095, "Each observation averages 3 correlated internal rates. Separate workload scales; no historical pooling.", fontsize=9.5, color=MUTED)
    fig.text(.07, .05, "Descriptive baselines using upstream placement functionality. No optimizer or causal speedup claim.", fontsize=9.5, color=MUTED)
    save(fig, output, "c1-baselines")


def overhead(data, output):
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    fig.subplots_adjust(left=.14, right=.96, top=.73, bottom=.32)
    fig.text(.07, .93, "Measured instrumentation overhead", fontsize=17, weight="bold")
    fig.text(.07, .875, "Six counterbalanced U/T process pairs · original runtime versus v3 tracing runtime", fontsize=10.5, color=MUTED)
    colors = {"UT": "#33658a", "TU": "#aa6147"}
    summaries = data["c2"]["overhead"]["primary_workloads"]
    for y, workload in enumerate(("pp512", "tg128")):
        pairs = [p for p in data["c2"]["overhead"]["pairs"] if p["workload"] == workload]
        for index, p in enumerate(pairs):
            ax.scatter(p["overhead_percent"], y + (index - 2.5) * .045, color=colors[p["order"]], s=35, alpha=.75, zorder=3)
        s = summaries[workload]
        mean = s["geometric_overhead_percent"]
        low, high = s["nominal_95pct_ci_percent"]
        ax.errorbar(mean, y + .25, xerr=[[mean - low], [high - mean]], fmt="D", color=INK, markersize=6, capsize=5, lw=1.6, zorder=4)
        ax.annotate(f"{mean:+.3f}%  [{low:+.3f}%, {high:+.3f}%]", (mean, y + .25), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=10, weight="bold")
    ax.axvline(0, color="#93a4ae", linestyle="--", lw=1)
    ax.set_yticks([0, 1], ["PP512", "TG128"])
    ax.set_ylim(-.25, 1.65)
    ax.set_xlim(-.75, 1.38)
    ax.set_xlabel("Elapsed-time overhead: 100 × (T / U − 1), percent", labelpad=10)
    ax.grid(axis="x", color="#e5eaee")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=colors[o], label=f"Pair: {o}") for o in ("UT", "TU")] + [Line2D([], [], marker="D", color=INK, label="Geometric mean + nominal 95% CI")], frameon=False, loc="upper left", bbox_to_anchor=(-.02, -.27), ncol=3, fontsize=9)
    fig.text(.07, .095, "Independent observations are paired processes; internal repetitions are correlated. Small n and order effects remain.", fontsize=9.3, color=MUTED)
    fig.text(.07, .05, "Intervals use Student t on six paired log ratios (df = 5). This calibrates the observer; it is not an optimization result.", fontsize=9.3, color=MUTED)
    save(fig, output, "c2-overhead")


def attribution(data, output):
    metrics = [
        ("phase_wall_ms", "Phase wall duration", "#172b3a"),
        ("ROCm0_graph_stream_ms_sum", "ROCm graph stream sum", "#aa6147"),
        ("CUDA0_graph_stream_ms_sum", "CUDA graph stream sum", "#33658a"),
        ("cross_fallback_cpu_union_ms", "Scheduler fallback CPU union", "#7a7398"),
        ("fallback_nested_sync_union_ms", "Nested producer-sync coverage", "#7a7398"),
        ("host_stage_cpu_union_ms", "Host-stage parent CPU union", "#208477"),
        ("host_stage_native_legs_stream_ms_sum", "Native copy-leg stream sum", "#208477"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.5), sharey=True)
    fig.subplots_adjust(left=.28, right=.98, top=.77, bottom=.30, wspace=.13)
    fig.text(.055, .94, "Phase-attributed CPU and device-stream measurements", fontsize=17, weight="bold")
    fig.text(.055, .89, "C3 offline attribution from six accepted instrumented processes · separate overlapping measurements", fontsize=10.5, color=MUTED)
    for ax, workload, title in zip(axes, ("pp512", "tg128"), ("PP512", "TG128")):
        for i, (metric, _, color) in enumerate(metrics):
            s = data["c3"]["metrics"][workload]["metrics"][metric]
            value, sd = s["process_mean"], s["between_process_sample_sd"]
            ax.errorbar(value, i, xerr=sd, fmt="o", color=color, markersize=6, capsize=4, lw=1.4)
            ax.annotate(f"{value:,.2f}", (value, i), xytext=(-9 if value >= 1000 else 7, 5), ha="right" if value >= 1000 else "left", textcoords="offset points", fontsize=9, color=color)
        ax.set_xscale("log")
        ax.set_xlim(1, 16000)
        ax.set_ylim(len(metrics) - .45, -.75)
        ax.set_yticks(range(len(metrics)), [m[1] for m in metrics])
        ax.set_xlabel("Duration (ms, logarithmic scale)", labelpad=9)
        ax.set_title(title, loc="left", pad=14, fontsize=12)
        ax.grid(axis="x", which="major", color="#e5eaee")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.text(.055, .16, "Dots: mean of six process means; whiskers: between-process sample SD. These rows overlap and must not be summed.", fontsize=10, weight="bold")
    fig.text(.055, .115, "Device stream intervals include dependencies/gaps, not kernel-only time. CPU synchronization is not wasted GPU time.", fontsize=9.5, color=MUTED)
    fig.text(.055, .07, "TG staging CPU calls cover 0.993% of phase wall duration; ROCm lies on a serialized dependency path.", fontsize=9.5, color=MUTED)
    fig.text(.055, .03, "The internal ROCm kernel/gap breakdown and an avoidable dominant bottleneck remain unknown.", fontsize=9.5, color=MUTED)
    save(fig, output, "c3-attribution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "evidence/portfolio/results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/portfolio")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    if data.get("schema") != "heterollm.portfolio.v1":
        raise SystemExit("Unsupported public evidence schema")
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    c1(data, args.output)
    overhead(data, args.output)
    attribution(data, args.output)
    print("Rendered three public evidence figures as SVG and PNG; no workload executed")


if __name__ == "__main__":
    main()
