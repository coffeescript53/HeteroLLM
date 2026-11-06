"""CPU-only tests of optional instrumentation; no GPU runtime or model loaded."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trace_patch", ROOT / "instrumentation/build_patch.py")
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)

HARNESS = r'''
#include "heterollm-cpu-trace.h"
#include <cstring>
#include <thread>
#include <vector>
int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "normal";
    if (!std::strcmp(mode, "disabled")) {
        int changed = 0;
        HT_TRACE_SCOPE("disabled", (++changed, ""), "", ++changed);
        HT_TRACE_GRAPH(++changed, ++changed); HT_TRACE_SPLIT(++changed);
        return changed != 0;
    }
#if defined(HETEROLLM_CPU_TRACE) && HETEROLLM_CPU_TRACE
    if (!std::strcmp(mode, "overflow")) {
        for (int i = 0; i < 18; ++i) { HT_TRACE_SCOPE("overflow", "CUDA0", "ROCm0", 8192); }
    } else if (!std::strcmp(mode, "unfinished")) {
        new heterollm_trace::Span("unfinished", "CUDA0", "", 0);
    } else if (!std::strcmp(mode, "truncated")) {
        HT_TRACE_SCOPE("long", "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz", "", 0);
    } else if (!std::strcmp(mode, "concurrent")) {
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) threads.emplace_back([] {
            HT_TRACE_GRAPH(4, 2); HT_TRACE_SPLIT(3);
            for (int i = 0; i < 40; ++i) { HT_TRACE_SCOPE("thread", "CUDA0", "ROCm0", 64); }
        });
        for (auto & thread : threads) thread.join();
    } else if (!std::strcmp(mode, "abrupt")) {
        { HT_TRACE_SCOPE("before_abrupt_exit", "", "", 0); }
        _exit(7);
    } else {
        HT_TRACE_GRAPH(4, 2); HT_TRACE_SPLIT(3);
        HT_TRACE_SCOPE("outer", "CUDA0", "ROCm0", 8192);
        { HT_TRACE_SCOPE("inner", "quote\"slash\\", "host_malloc", 8192); }
    }
#endif
}
'''


class TraceCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("c++"):
            raise unittest.SkipTest("candidate trace tests require a CPU C++ compiler")
        cls.temp = tempfile.TemporaryDirectory(prefix="heterollm-trace-cpu-")
        cls.directory = Path(cls.temp.name)
        source = cls.directory / "harness.cpp"
        source.write_text(HARNESS)
        cls.binaries = {}
        for variant, definitions in {"disabled": [], "enabled": ["-DHETEROLLM_CPU_TRACE=1"],
                                     "small": ["-DHETEROLLM_CPU_TRACE=1", "-DHETEROLLM_TRACE_CAPACITY=16"]}.items():
            binary = cls.directory / variant
            subprocess.run(["c++", "-std=c++11", "-O2", "-pthread", *definitions,
                            "-I", str(ROOT / "instrumentation"), str(source), "-o", str(binary)],
                           check=True, capture_output=True, text=True, timeout=30)
            cls.binaries[variant] = binary

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_trace(self, mode="normal", variant="enabled"):
        output = self.directory / (self.id().split('.')[-1] + "-" + mode + ".jsonl")
        env = dict(os.environ, HETEROLLM_TRACE_FILE=str(output))
        result = subprocess.run([str(self.binaries[variant]), mode], env=env, text=True,
                                capture_output=True, timeout=10)
        rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
        return result, rows, output

    def test_disabled_does_not_evaluate_arguments_or_create_file(self):
        result, rows, path = self.run_trace("disabled", "disabled")
        self.assertEqual(result.returncode, 0)
        self.assertFalse(path.exists())

    def test_nested_duration_context_and_escaped_names(self):
        result, rows, _ = self.run_trace()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(rows[0]["time_domain"], "cpu")
        self.assertEqual(rows[0]["phase"], "unknown")
        spans = rows[1:-1]
        self.assertEqual([r["kind"] for r in spans], ["outer", "inner"])
        self.assertTrue(rows[-1]["complete"])
        self.assertLessEqual(spans[0]["begin_ns"], spans[1]["begin_ns"])
        self.assertGreaterEqual(spans[0]["end_ns"], spans[1]["end_ns"])
        self.assertEqual(spans[1]["src"], 'quote"slash\\')
        for event in spans:
            self.assertGreaterEqual(event["end_ns"], event["begin_ns"])
            self.assertEqual((event["graph"], event["split"], event["copy"], event["n_copies"]), (1, 3, 2, 4))
            self.assertEqual(event["logical_bytes"], 8192)

    def test_overflow_is_bounded_and_explicitly_incomplete(self):
        result, rows, _ = self.run_trace("overflow", "small")
        self.assertEqual(len(rows), 18)
        self.assertEqual(rows[-1]["recorded"], 16)
        self.assertEqual(rows[-1]["dropped"], 2)
        self.assertFalse(rows[-1]["complete"])
        self.assertIn("HETEROLLM_TRACE_ERROR", result.stderr)

    def test_unfinished_and_truncated_are_incomplete(self):
        for mode, field in [("unfinished", "unfinished"), ("truncated", "metadata_truncated")]:
            _, rows, _ = self.run_trace(mode)
            self.assertEqual(rows[-1][field], 1)
            self.assertFalse(rows[-1]["complete"])

    def test_concurrent_calls_keep_unique_order_and_thread_context(self):
        _, rows, _ = self.run_trace("concurrent")
        spans = rows[1:-1]
        self.assertEqual(len(spans), 160)
        self.assertEqual([e["seq"] for e in spans], list(range(160)))
        self.assertEqual([e["begin_ns"] for e in spans], sorted(e["begin_ns"] for e in spans))
        self.assertEqual(len({e["graph"] for e in spans}), 4)
        self.assertTrue(rows[-1]["complete"])

    def test_output_collision_never_overwrites_and_abrupt_exit_is_incomplete(self):
        result, rows, output = self.run_trace()
        before = output.read_bytes()
        repeated = subprocess.run([str(self.binaries["enabled"])],
                                  env=dict(os.environ, HETEROLLM_TRACE_FILE=str(output)),
                                  capture_output=True, text=True, timeout=10)
        self.assertEqual(output.read_bytes(), before)
        self.assertIn("fresh output open failed", repeated.stderr)
        result, rows, _ = self.run_trace("abrupt")
        self.assertEqual(result.returncode, 7)
        self.assertEqual([r["type"] for r in rows], ["header"])

    def test_patch_reproducible_pinned_identity_and_applies_cleanly(self):
        self.assertEqual(PATCH.make_patch(), (ROOT / "instrumentation/ggml-cpu-trace.patch").read_text())
        original = (ROOT / "llama.cpp/ggml/src/ggml-backend.cpp").read_text()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            PATCH.patched_source(original + "\n")
        copy = self.directory / "apply-check"
        (copy / "ggml/src").mkdir(parents=True)
        (copy / "ggml/src/ggml-backend.cpp").write_text(original)
        subprocess.run(["git", "apply", "--check", str(ROOT / "instrumentation/ggml-cpu-trace.patch")],
                       cwd=copy, check=True, capture_output=True, timeout=10)
        subprocess.run(["git", "apply", str(ROOT / "instrumentation/ggml-cpu-trace.patch")],
                       cwd=copy, check=True, capture_output=True, timeout=10)
        self.assertEqual((copy / "ggml/src/ggml-backend.cpp").read_text(), PATCH.patched_source(original))
        # Syntax-check the complete actual translation unit, enabled and disabled.
        for definitions in [[], ["-DHETEROLLM_CPU_TRACE=1"]]:
            subprocess.run(["c++", "-std=c++17", "-fsyntax-only", "-pthread", *definitions,
                            "-I", str(ROOT / "llama.cpp/ggml/include"),
                            "-I", str(ROOT / "llama.cpp/ggml/src"),
                            str(copy / "ggml/src/ggml-backend.cpp")],
                           check=True, capture_output=True, text=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
