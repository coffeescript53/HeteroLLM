#!/usr/bin/env python3
"""Export an allowlisted public view of already accepted, immutable analyses.

This does not execute a workload or rerun the original analysis. Every input is
bound to its archived SHA-256. Repository-relative paths resolve from --root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "c1_performance": ("stage-c1-memtotal-execution-20261025-zazshxhs/PERFORMANCE_SUMMARY.json", "91f3e24e6b6fd7acc00c10a7deb014ec1b262c5800ac42df154bcb138284238e"),
    "c1_resources": ("stage-c1-memtotal-execution-20261025-zazshxhs/RESOURCE_SUMMARY.json", "1262f1b9dab370e333364f7d68a525f2778ef089c73680d355eb3399d206e3e9"),
    "c1_placement": ("stage-c1-memtotal-execution-20261025-zazshxhs/PLACEMENT_EVIDENCE.json", "324fe98dcb3a68b33bc2a6dd10a6c922ef320ed6075eb9d50578a334d008fea0"),
    "c2_equivalence": ("c2-equivalence-execution-review-20261027/OFFLINE_READBACK_VALIDATION.json", "568382245bb638ae321ed380e4f9ac80374011fa4b6960ba9236263483135371"),
    "c2_overhead": ("c2-overhead-execution-review-20261027/INDEPENDENT_STATISTICAL_REPLAY.json", "36434884fe9fdb090673e61bd07cafcebe9d3e1c56c614f3277fa01a201766d8"),
    "c2_pairs": ("c2-overhead-execution-review-20261027/PAIRED_RESULTS.csv", "f9a39fc3079d7bcfd5a19878f446af448ff02ca70eeef1ec6a5bc860a7ae8e80"),
    "c3_analysis": ("c3-attribution-preparation-20261030/ARCHIVED_TRACE_ANALYSIS.json", "afeb7b1595cf64b6e51efa03abeaf4388d0eeb74266c6d41498b5350449c960c"),
    "c3_processes": ("c3-attribution-preparation-20261030/ARCHIVED_PROCESS_TABLE.csv", "a578079424e20876c12c680fc744e1698f8b52ac66e152b8bc8cab1e5e3a1b95"),
}
METRICS = (
    "phase_wall_ms", "host_stage_cpu_union_ms",
    "host_stage_native_legs_stream_ms_sum", "cross_fallback_cpu_union_ms",
    "fallback_nested_sync_union_ms", "fallback_uncovered_cpu_ms",
    "backend_sync_cpu_union_ms", "graph_split_allocate_cpu_union_ms",
    "graph_cpu_submit_union_ms", "CUDA0_graph_stream_ms_sum",
    "ROCm0_graph_stream_ms_sum", "host_stage_pct_of_phase",
)
LABELS = {"A": "RTX 5090 + RAM", "B": "R9700 + RAM", "D": "Default mixed", "E": "Equal mixed"}
# Expected hashes copied from the original C1 PACKAGE_MANIFEST.json and C2
# RAW_EVIDENCE_INVENTORY.json, not adopted from a new hash of the stdout files.
SAMPLE_SOURCES = {
    "c1": ("stage-c1-memtotal-fixed-timed-20261025", [
        ("01-r1-mixed-equal", "d7327b08d917f00c70dc7830a605d3cd17eade827b04fd956bc6f22e22d46e22"),
        ("02-r1-mixed-default", "0f138f3fe27d88008aa413a0be163178506cdf06f4964590d5b707993942d412"),
        ("03-r1-cuda-only", "4a308ffd10627cbbf3ae3e98068821d1f0694d62b7130f562ee36836ba503a47"),
        ("04-r1-rocm-only", "ebd9c070f6848ea6ed8c4628e4b2a37ed319ea11e7de3095fcfa2770fbcac3e6"),
        ("05-r2-mixed-default", "059f1f62807b75bd73b5d914fc9c011811492db1cb888d381749638f1640b48c"),
        ("06-r2-rocm-only", "906b93ed36c385a83595b794a4ca0b83fa1cb966a3eea53246f45f8522f1a011"),
        ("07-r2-mixed-equal", "05d9efff8f4845b0e5d4435ef55d4d0138fad7c4891088882bb2ad24a2c0d1cd"),
        ("08-r2-cuda-only", "747b26ef2649b714db492cee9deebd79517ad535d074d4a420b5712fda9dc38d"),
        ("09-r3-rocm-only", "5c54bde5da5826472889735bfd8f4358610c96e98085588a01186d74786e2eb5"),
        ("10-r3-cuda-only", "9a3cfb3a9b33f57e46658b2facd32c798eeb5d4c94aed1d6730a0f7e08d66c01"),
        ("11-r3-mixed-default", "70a8fed551cc0be7f6cd7ad9added6310a7835b985bf291cb2d54ba6073eeed3"),
        ("12-r3-mixed-equal", "2080178454d472c67312f41683867ebb492b9810eee603a43d86ac2973fbbeb0"),
    ]),
    "c2_overhead": ("stage-c2-overhead-20261027", [
        ("01-block1-U", "973b9791f54c5fb9e2090c868376f49f0378ae7f3dd4fd7835a6ef1b05201765"),
        ("02-block1-T", "80246d156350317877570417f3e82dffaad96cf01be49304e58b6916627ef747"),
        ("03-block2-T", "4798072b2e19137922f327cdbd183f0f3cf1dab0d8ffe527795b70acc0243ca5"),
        ("04-block2-U", "7f3593f9f868a77a4249c1437aab1f3f4e7238f500796a38acc657435ee09aad"),
        ("05-block3-U", "9c1e6d876020418aed22121b69ebd932388d39ac9c8c39a5d6290e43aaa574ee"),
        ("06-block3-T", "73a6faaf466c15064796ad8f8b7670ee6bebb47f3f470208f81bcc7a3c42290f"),
        ("07-block4-T", "b1e25b2f2914fb1485853690484bd143d1782d5a44d5964d53f525ce6e75812f"),
        ("08-block4-U", "f60ed4366230a53841761357c8c40d8f8acf62e8b759c094961a77db184f3bd9"),
        ("09-block5-U", "9611fdd68631916b2f0738f6468a639c5459a13db090bd4d66c900fdaaae3923"),
        ("10-block5-T", "eb2d7e4f46bdea35f49a7bb1718d8a1917c70b85ac7f3dcea3bfbf749074aa08"),
        ("11-block6-T", "f27decc4586487c800cad9fab9605ec7f392004f76d4de21fab1896723070ccf"),
        ("12-block6-U", "9caf70c3189967bb6cb4809a041f19848abb0d3b7f4c598dab033f1911386f3a"),
    ]),
}


def extract_timings(root: Path) -> dict:
    datasets = {}
    provenance = {}
    for dataset, (archive, sources) in SAMPLE_SOURCES.items():
        processes = []
        for name, digest in sources:
            raw = verified_bytes(root / "runs/llama33-70b" / archive / name / "stdout.txt", digest)
            rows = json.loads(raw)
            ordinal = int(name.split("-")[0])
            process_id = f"{dataset}-{ordinal:02d}"
            process = {"process": process_id, "ordinal": ordinal, "workloads": {}}
            if dataset == "c1":
                process["round"] = int(name.split("-")[1][1:])
                configuration = "-".join(name.split("-")[2:])
                process["configuration"] = {"cuda-only": "A", "rocm-only": "B", "mixed-default": "D", "mixed-equal": "E"}[configuration]
            else:
                process["block"] = int(name.split("-")[1][5:])
                process["arm"] = name.split("-")[2]
            for row in rows:
                identity = (row["n_prompt"], row["n_gen"], row["n_depth"])
                workload = {(512, 0, 0): "pp512", (0, 128, 0): "tg128"}.get(identity)
                if workload is None or workload in process["workloads"] or len(row["samples_ns"]) != 3:
                    raise ValueError("Unexpected original timed workload structure")
                if any(type(x) is not int or x <= 0 for x in row["samples_ns"]):
                    raise ValueError("Original duration must be a positive integer nanosecond count")
                process["workloads"][workload] = {k: row[k] for k in ("n_prompt", "n_gen", "n_depth", "samples_ns")}
            if set(process["workloads"]) != {"pp512", "tg128"}:
                raise ValueError("Missing original timed workload")
            processes.append(process)
            provenance[process_id] = {"archive": archive, "file": f"{name}/stdout.txt", "sha256": digest, "bytes": len(raw)}
        datasets[dataset] = {"independent_processes": 12, "correlated_internal_samples": 72, "processes": processes}
    return {"schema": "heterollm.portfolio.raw-timings.v1", "scope": "Allowlisted original integer samples_ns; C1 and contemporaneous C2 U/T remain separate datasets. Each process contains two workloads with three correlated repetitions.", "duration_unit": "nanoseconds", "source_count": 24, "datasets": datasets, "provenance": provenance, "expected_hash_origin": {"c1": "C1 accepted PACKAGE_MANIFEST.json", "c2_overhead": "C2 accepted RAW_EVIDENCE_INVENTORY.json"}}


def verified_bytes(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Archived input hash mismatch: {path.name}")
    return data


def extract(root: Path) -> dict:
    inputs = {}
    provenance = {}
    for key, (relative, digest) in SOURCES.items():
        data = verified_bytes(root / "runs" / "llama33-70b" / relative, digest)
        inputs[key] = json.loads(data) if relative.endswith(".json") else list(csv.DictReader(io.StringIO(data.decode())))
        provenance[key] = {"archive": relative.split("/")[0], "file": Path(relative).name, "sha256": digest, "bytes": len(data)}

    performance = inputs["c1_performance"]
    groups = {}
    for key, label in LABELS.items():
        original = performance["groups"][key]
        placements = [x for x in inputs["c1_placement"]["launches"] if x["id"] == key]
        layer_map = placements[0]["c0_observed_layer_map"]
        if any(x["c0_observed_layer_map"] != layer_map for x in placements):
            raise ValueError("Reference placement changed across C1 launches")
        groups[key] = {
            "label": label,
            "pp512": original["pp"], "tg128": original["tg"],
            "observations": [{k: row[k] for k in ("ordinal", "round", "pp_tokens_s", "tg_tokens_s")} for row in original["observations"]],
            "reference_layer_map": layer_map,
            "resources": inputs["c1_resources"]["groups"][key],
        }

    numerical = inputs["c2_equivalence"]
    numerical_keys = ("status", "atol", "rtol", "reference", "bitwise_identical_logits", "rows", "vocabulary", "finite_float32_values_per_arm", "raw_bytes_per_arm", "raw_logits_sha256", "outside_tolerance", "different_values", "maximum_absolute_difference", "rmse", "exact_greedy_id_agreements", "exact_piece_byte_agreements")
    pairs = []
    for row in inputs["c2_pairs"]:
        pairs.append({"block": int(row["block"]), "order": row["order"], "workload": row["workload"], **{k: float(v) for k, v in row.items() if k not in {"block", "order", "workload"}}})

    statistics = inputs["c3_analysis"]["six_process_statistics"]
    selected = {workload: {"independent_T_processes": entry["independent_T_processes"], "correlated_internal_repetitions_total": entry["correlated_internal_repetitions_total"], "metrics": {k: entry["metrics"][k] for k in METRICS}} for workload, entry in statistics.items()}
    process_ids = {}
    process_rows = []
    for row in inputs["c3_processes"]:
        process_id = process_ids.setdefault(row["run"], f"T{len(process_ids) + 1}")
        process_rows.append({"process": process_id, "workload": row["workload"], "internal_repetitions_correlated": int(row["internal_repetitions_correlated"]), **{k: float(row[k]) for k in METRICS}})
    routing = {}
    for workload in ("pp512", "tg128"):
        phases = [p for p in inputs["c3_analysis"]["phases"] if p["run"] in process_ids and p["phase"] == workload]
        counts = {p["host_stage_count"] for p in phases}
        payloads = {p["host_stage_logical_bytes"] for p in phases}
        if len(phases) != 18 or len(counts) != 1 or len(payloads) != 1:
            raise ValueError("Unexpected accepted routing repetition structure")
        routing[workload] = {
            "direction": "CUDA0 -> host -> ROCm0",
            "measured_repetitions": len(phases),
            "parents_per_repetition": counts.pop(),
            "logical_bytes_per_repetition": payloads.pop(),
            "source_complete_then_stage_then_destination_submit": sum(p["source_completion_before_staging_before_destination_submit_count"] for p in phases),
            "adjacent_graph_pairs": sum(p["consecutive_graph_pairs"] for p in phases),
            "prior_rocm_complete_before_next_cuda_enqueue": sum(p["prior_rocm_completion_before_next_cuda_enqueue_count"] for p in phases),
            "graph_kinds_per_repetition": phases[0]["graph_kinds"],
        }
    return {
        "schema": "heterollm.portfolio.v1",
        "scope": "Public allowlisted projection of accepted C1/C2/C3 analyses; no new GPU experiment or rewritten historical result.",
        "review_status": "Owner reports independent ChatGPT acceptance of the original C1/C2/C3 evidence. This C4 public projection requires its own review.",
        "units": {"throughput": "tokens/second", "duration": "milliseconds unless field says ns", "memory": "bytes; GiB = bytes / 2**30", "temperature": "degrees Celsius", "percent": "percent; SD of percentages is percentage points"},
        "identity": {"model": "Llama 3.3 70B Instruct Q4_K_M", "model_bytes": 42520398816, "model_sha256": "32df3baccb556f9840059b2528b2dee4d3d516b24afdfb9d0c56ff5f63e3a664", "llama_cpp_revision": "5e03bdd8700948b9c41c54dd1b00f28a2aebc03f", "cpu": "Ryzen 9 9950X", "host_ram_nominal_gb": 96, "gpus": ["RTX 5090 32 GB (CUDA0)", "Radeon AI PRO R9700 32 GB (ROCm0)"]},
        "c1": {"method": performance["method"], "claims": performance["claims"], "independent_processes": 12, "correlated_internal_samples": 72, "launch_order": ["E", "D", "A", "B", "D", "B", "E", "A", "B", "A", "D", "E"], "placement_scope": "C0 observed reference map corroborated by quiet C1 allocation/activity; not a new per-layer trace. Indices 0-79 are transformer blocks; 80 is output.", "resource_scope": "Sampled maxima across each configuration's three processes except wait4 lifetime RSS; VRAM includes system-wide background allocations; system nonavailable memory is not process RSS.", "groups": groups},
        "c2": {"correctness": {**{k: numerical[k] for k in numerical_keys}, "scope": "One predeclared teacher-forced fixture, exact runtimes and settings; not general model correctness or arbitrary-generation equivalence."}, "overhead": {"independent_pairs": 6, "internal_repetitions_per_process": 3, "method": inputs["c2_overhead"]["method"], "limitations": inputs["c2_overhead"]["limitations"], "primary_workloads": inputs["c2_overhead"]["primary_workloads"], "pairs": pairs}},
        "c3": {"method": "Means and sample SD across six T process means, with three correlated repetitions within each process. The earlier single diagnostic is excluded.", "metrics": selected, "processes": process_rows, "routing": routing, "interpretation": "A longer ROCm graph region lies on the serialized TG dependency path; no removable kernel bottleneck or implementation-ready optimization is established.", "limitations": ["Overlapping inclusive scopes must not be summed into a wall-time budget.", "Same-device stream intervals include dependencies and gaps; they are not kernel-only time.", "CPU synchronization duration is not wasted GPU time.", "Logical tensor bytes are not measured physical PCIe traffic.", "Independent CUDA/HIP clock timestamps are not subtracted; precise cross-device overlap is unknown.", "Native copy-leg durations are not added to their inclusive host-stage parent duration.", "Instrumented observations are not corrected by subtracting average instrumentation overhead."]},
        "provenance": provenance,
    }


def encoded(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/portfolio/results.json")
    parser.add_argument("--timings-output", type=Path, default=ROOT / "evidence/portfolio/raw-timings.json")
    parser.add_argument("--check", action="store_true", help="Verify existing projection without writing")
    args = parser.parse_args()
    outputs = {args.output: encoded(extract(args.root)), args.timings_output: encoded(extract_timings(args.root))}
    for path, result in outputs.items():
        if args.check:
            if path.read_text() != result:
                raise SystemExit("Public projection differs from the hash-bound accepted inputs")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result)
            if path.read_text() != result:
                raise OSError("Public projection write readback mismatch")
    print("Verified public projections against eight archived analyses and 24 original stdout files (144 duration samples)")


if __name__ == "__main__":
    main()
