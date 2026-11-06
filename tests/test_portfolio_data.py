"""CPU-only checks for the C4 public projection; no runtime/device imports."""
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("derive_portfolio", ROOT / "scripts/derive-portfolio-data.py")
DERIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DERIVE)


class PortfolioDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "evidence/portfolio/results.json").read_text())
        cls.timings = json.loads((ROOT / "evidence/portfolio/raw-timings.json").read_text())

    def test_c1_independent_observations_and_descriptive_statistics(self):
        c1 = self.data["c1"]
        self.assertEqual(c1["independent_processes"], 12)
        self.assertEqual(c1["correlated_internal_samples"], 72)
        observed_order = {}
        for key, group in c1["groups"].items():
            self.assertEqual(len(group["observations"]), 3)
            for row in group["observations"]:
                observed_order[row["ordinal"]] = key
            for workload, field in (("pp512", "pp_tokens_s"), ("tg128", "tg_tokens_s")):
                samples = [x[field] for x in group["observations"]]
                self.assertEqual(group[workload]["independent_launch_means"], samples)
                self.assertEqual(group[workload]["independent_n"], 3)
                self.assertAlmostEqual(group[workload]["mean"], statistics.mean(samples), places=12)
                self.assertAlmostEqual(group[workload]["sample_sd"], statistics.stdev(samples), places=12)
        self.assertEqual([observed_order[k] for k in range(1, 13)], c1["launch_order"])

    def test_placement_keeps_output_separate_and_actual_equal_boundary(self):
        expected = {"A": [(0, 24, "CPU"), (25, 80, "CUDA0")], "B": [(0, 21, "CPU"), (22, 80, "ROCm0")], "D": [(0, 39, "CUDA0"), (40, 80, "ROCm0")], "E": [(0, 40, "CUDA0"), (41, 80, "ROCm0")]}
        for key, spans in expected.items():
            actual = self.data["c1"]["groups"][key]["reference_layer_map"]
            self.assertEqual(actual, {str(i): backend for start, stop, backend in spans for i in range(start, stop + 1)})
        self.assertIn("not a new per-layer trace", self.data["c1"]["placement_scope"])

    def test_paired_overhead_preserves_estimand_and_uncertainty(self):
        overhead = self.data["c2"]["overhead"]
        for workload, summary in overhead["primary_workloads"].items():
            pairs = [x for x in overhead["pairs"] if x["workload"] == workload]
            self.assertEqual([x["order"] for x in pairs], ["UT", "TU"] * 3)
            ratios = [x["T_mean_elapsed_ns"] / x["U_mean_elapsed_ns"] for x in pairs]
            for pair, ratio in zip(pairs, ratios):
                self.assertEqual(pair["T_over_U_elapsed_ratio"], ratio)
            logs = list(map(math.log, ratios))
            mu = statistics.mean(logs)
            radius = 2.570581835636305 * statistics.stdev(logs) / math.sqrt(6)
            self.assertAlmostEqual(summary["geometric_overhead_percent"], 100 * math.expm1(mu), places=11)
            for actual, expected in zip(summary["nominal_95pct_ci_percent"], [100 * math.expm1(mu - radius), 100 * math.expm1(mu + radius)]):
                self.assertAlmostEqual(actual, expected, places=11)
        self.assertLess(overhead["primary_workloads"]["pp512"]["nominal_95pct_ci_percent"][0], 0)

    def test_correctness_is_bounded_to_fixed_fixture(self):
        c = self.data["c2"]["correctness"]
        self.assertEqual(c["rows"] * c["vocabulary"], 16545024)
        self.assertEqual(c["raw_bytes_per_arm"], 4 * c["rows"] * c["vocabulary"])
        self.assertEqual(c["raw_logits_sha256"]["U"], c["raw_logits_sha256"]["T"])
        self.assertEqual((c["atol"], c["rtol"]), (1e-4, 1e-5))
        self.assertEqual((c["outside_tolerance"], c["different_values"]), (0, 0))
        self.assertEqual((c["exact_greedy_id_agreements"], c["exact_piece_byte_agreements"]), (129, 129))
        self.assertIn("not general model correctness", c["scope"])

    def test_c3_statistics_use_six_processes_not_eighteen_repetitions(self):
        c3 = self.data["c3"]
        for workload, group in c3["metrics"].items():
            self.assertEqual(group["independent_T_processes"], 6)
            self.assertEqual(group["correlated_internal_repetitions_total"], 18)
            rows = [x for x in c3["processes"] if x["workload"] == workload]
            self.assertEqual(len({x["process"] for x in rows}), 6)
            for metric, summary in group["metrics"].items():
                values = [x[metric] for x in rows]
                self.assertAlmostEqual(summary["process_mean"], statistics.mean(values), places=10)
                self.assertAlmostEqual(summary["between_process_sample_sd"], statistics.stdev(values), places=10)

    def test_copy_counts_units_and_unknown_ordering_remain_explicit(self):
        routing = self.data["c3"]["routing"]
        self.assertEqual((routing["pp512"]["parents_per_repetition"], routing["pp512"]["logical_bytes_per_repetition"]), (4, 16 * 2**20))
        self.assertEqual((routing["tg128"]["parents_per_repetition"], routing["tg128"]["logical_bytes_per_repetition"]), (128, 4 * 2**20))
        self.assertEqual(routing["tg128"]["source_complete_then_stage_then_destination_submit"], 2304)
        self.assertEqual(routing["tg128"]["prior_rocm_complete_before_next_cuda_enqueue"], 2286)
        self.assertEqual(routing["pp512"]["adjacent_graph_pairs"] - routing["pp512"]["prior_rocm_complete_before_next_cuda_enqueue"], 36)
        limits = " ".join(self.data["c3"]["limitations"])
        for phrase in ("must not be summed", "not kernel-only", "not wasted GPU time", "not measured physical PCIe traffic", "overlap is unknown"):
            self.assertIn(phrase, limits)

    def test_public_projection_contains_no_host_identifiers_or_private_urls(self):
        encoded = json.dumps([self.data, self.timings])
        for pattern in (r"/home/", r"/tmp/", r"/proc/", r"/sys/", r"drive\.google\.com", r"docs\.google\.com", r"\b0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\b", r'"pid"', r"dimm-00"):
            self.assertIsNone(re.search(pattern, encoded, re.IGNORECASE), pattern)
        for item in self.data["provenance"].values():
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(item["bytes"], 0)
        self.assertIn("2**30", self.data["units"]["memory"])

    def test_changed_archived_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                DERIVE.verified_bytes(path, "0" * 64)

    def test_public_view_matches_hash_bound_originals_when_available(self):
        inputs = [ROOT / "runs/llama33-70b" / relative for relative, _ in DERIVE.SOURCES.values()]
        inputs += [ROOT / "runs/llama33-70b" / archive / name / "stdout.txt" for archive, sources in DERIVE.SAMPLE_SOURCES.values() for name, _ in sources]
        if not all(path.exists() for path in inputs):
            self.skipTest("Private frozen research archive is not included in the public source release")
        self.assertEqual(DERIVE.extract(ROOT), self.data)
        self.assertEqual(DERIVE.extract_timings(ROOT), self.timings)

    def test_original_c1_integer_durations_reproduce_process_rates(self):
        dataset = self.timings["datasets"]["c1"]
        self.assertEqual(dataset["independent_processes"], 12)
        self.assertEqual(sum(len(w["samples_ns"]) for p in dataset["processes"] for w in p["workloads"].values()), 72)
        for process in dataset["processes"]:
            observation = next(x for x in self.data["c1"]["groups"][process["configuration"]]["observations"] if x["ordinal"] == process["ordinal"])
            self.assertEqual(observation["round"], process["round"])
            for workload, fields in process["workloads"].items():
                count = fields["n_prompt"] + fields["n_gen"]
                self.assertEqual(fields["n_depth"], 0)
                self.assertEqual((fields["n_prompt"], fields["n_gen"]), (512, 0) if workload == "pp512" else (0, 128))
                self.assertTrue(all(type(n) is int and n > 0 for n in fields["samples_ns"]))
                rate = statistics.mean(count * 1e9 / n for n in fields["samples_ns"])
                self.assertAlmostEqual(rate, observation["pp_tokens_s" if workload == "pp512" else "tg_tokens_s"], places=11)

    def test_original_c2_integer_durations_reproduce_elapsed_pairs(self):
        dataset = self.timings["datasets"]["c2_overhead"]
        self.assertEqual(len(dataset["processes"]), 12)
        self.assertEqual(sum(len(w["samples_ns"]) for p in dataset["processes"] for w in p["workloads"].values()), 72)
        lookup = {(p["block"], p["arm"]): p for p in dataset["processes"]}
        for pair in self.data["c2"]["overhead"]["pairs"]:
            for arm in ("U", "T"):
                fields = lookup[pair["block"], arm]["workloads"][pair["workload"]]
                self.assertEqual((fields["n_prompt"], fields["n_gen"], fields["n_depth"]), (512, 0, 0) if pair["workload"] == "pp512" else (0, 128, 0))
                self.assertTrue(all(type(n) is int and n > 0 for n in fields["samples_ns"]))
                self.assertEqual(statistics.mean(fields["samples_ns"]), pair[f"{arm}_mean_elapsed_ns"])
        self.assertEqual(len(self.timings["provenance"]), 24)
        self.assertEqual(self.timings["duration_unit"], "nanoseconds")


if __name__ == "__main__":
    unittest.main()
