"""Same actual backend copy/data tests against the isolated v2 patch, CPU only."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from test_c2_fake_backend import FakeBackendTests as OriginalTests
from helpers import ROOT
from heterollm.trace_analysis import analyze


class FakeBackendV2Tests(OriginalTests):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='c2-v2-copy-');cls.path=Path(cls.temp.name)
        spec=importlib.util.spec_from_file_location('v2patch',ROOT/'instrumentation/v2/build_patch.py')
        patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)
        source=cls.path/'ggml-backend.cpp'
        source.write_text(patcher.patched_source((ROOT/'llama.cpp/ggml/src/ggml-backend.cpp').read_text()))
        (cls.path/'heterollm-cpu-trace.h').write_bytes((ROOT/'instrumentation/v2/heterollm-cpu-trace.h').read_bytes())
        cls.binary=cls.path/'fake-backend';cls.original=cls.path/'original-backend';cls.disabled=cls.path/'disabled-backend'
        for implementation,binary,flags in [(source,cls.binary,['-DHETEROLLM_CPU_TRACE=1']),
            (source,cls.disabled,[]),(ROOT/'llama.cpp/ggml/src/ggml-backend.cpp',cls.original,[])]:
            subprocess.run(['c++','-std=c++17','-O1','-pthread',*flags,
                '-DHETEROLLM_PATCHED_SOURCE="'+str(implementation)+'"','-I',str(ROOT/'llama.cpp/ggml/include'),
                '-I',str(ROOT/'llama.cpp/ggml/src'),str(ROOT/'tests/fixtures/c2_fake_backend.cpp'),
                '-L',str(ROOT/'llama.cpp/build-mixed/bin'),'-Wl,-rpath,'+str(ROOT/'llama.cpp/build-mixed/bin'),
                '-lggml-base','-o',str(binary)],check=True,capture_output=True,text=True,timeout=45)

    def test_90000_actual_event_callbacks_preserve_copies_order_and_data(self):
        trace=self.path/'flood.jsonl'
        runs=[subprocess.run([str(binary),'flood'],env={**os.environ,'HETEROLLM_TRACE_FILE':str(trace) if binary==self.binary else str(self.path/'disabled.jsonl')},capture_output=True,text=True,timeout=10)
              for binary in (self.binary,self.disabled,self.original)]
        self.assertTrue(all(r.returncode==0 for r in runs),[r.stderr for r in runs])
        self.assertEqual(runs[0].stdout,runs[1].stdout);self.assertEqual(runs[0].stdout,runs[2].stdout)
        self.assertFalse((self.path/'disabled.jsonl').exists())
        summary=analyze(trace.read_text(),runs[0].stderr)
        self.assertGreater(summary['aggregate_calls'],90000)
        self.assertEqual(summary['host_staged_directions']['CUDA0 -> ROCm0']['copies'],10)
        self.assertEqual(summary['host_staged_directions']['CUDA0 -> ROCm0']['logical_bytes'],160)
        self.assertEqual(summary['kinds']['copy.public_async_fallback']['count'],9)
        self.assertEqual(summary['kinds']['copy.scheduler_fallback']['count'],1)
        self.assertEqual(summary['kinds']['graph.submit']['count'],10)
        self.assertEqual(summary['aggregates']['event.record_submit | ROCm0']['count'],45001)
        self.assertEqual(summary['aggregates']['event.synchronize | ROCm0']['count'],45001)


# Do not collect the imported historical TestCase twice.
del OriginalTests
