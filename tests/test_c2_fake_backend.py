"""Actual pinned patched copy paths with CPU memory/callback mocks, no GPU APIs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from helpers import ROOT
from heterollm.trace_analysis import analyze


class FakeBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='c2-copy-cpu-');cls.path=Path(cls.temp.name)
        spec=importlib.util.spec_from_file_location('patcher',ROOT/'instrumentation/build_patch.py')
        patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)
        source=cls.path/'ggml-backend.cpp'
        source.write_text(patcher.patched_source((ROOT/'llama.cpp/ggml/src/ggml-backend.cpp').read_text()))
        (cls.path/'heterollm-cpu-trace.h').write_bytes((ROOT/'instrumentation/heterollm-cpu-trace.h').read_bytes())
        cls.binary=cls.path/'fake-backend';cls.original=cls.path/'original-backend'
        for implementation,binary in [(source,cls.binary),(ROOT/'llama.cpp/ggml/src/ggml-backend.cpp',cls.original)]:
            subprocess.run(['c++','-std=c++17','-O1','-pthread','-DHETEROLLM_CPU_TRACE=1',
                '-DHETEROLLM_PATCHED_SOURCE="'+str(implementation)+'"','-I',str(ROOT/'llama.cpp/ggml/include'),
                '-I',str(ROOT/'llama.cpp/ggml/src'),str(ROOT/'tests/fixtures/c2_fake_backend.cpp'),
                '-L',str(ROOT/'llama.cpp/build-mixed/bin'),'-Wl,-rpath,'+str(ROOT/'llama.cpp/build-mixed/bin'),
                '-lggml-base','-o',str(binary)],check=True,capture_output=True,text=True,timeout=45)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def check(self,mode,calls,required=(),absent=()):
        path=self.path/(mode+'.jsonl')
        run=subprocess.run([str(self.binary),mode],env={**os.environ,'HETEROLLM_TRACE_FILE':str(path)},
                           capture_output=True,text=True,timeout=5)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(run.stdout.splitlines(),calls)
        baseline=subprocess.run([str(self.original),mode],capture_output=True,text=True,timeout=5)
        self.assertEqual(baseline.returncode,0,baseline.stderr)
        self.assertEqual(run.stdout,baseline.stdout,'instrumentation changed callback ordering')
        kinds=analyze(path.read_text(),run.stderr)['kinds'] if path.exists() else {}
        for kind in required:self.assertIn(kind,kinds)
        for kind in absent:self.assertNotIn(kind,kinds)

    def test_scheduler_async_accepted_copies_before_graph_and_records_event(self):
        self.check('accepted',['event_wait','async_try','graph','event_record'],
                   ['copy.scheduler_async_accepted','graph.submit'],['copy.host_stage','copy.scheduler_fallback'])

    def test_scheduler_false_and_missing_async_preserve_sync_copy_event_order(self):
        for mode,attempt in [('fallback',['async_try']),('missing_callback',[])]:
            self.check(mode,['event_wait',*attempt,'sync:CUDA0','event_sync','buffer_try','get','set','graph','event_record'],
                       ['copy.scheduler_fallback','copy.host_stage','copy.host_stage.get','copy.host_stage.set'])

    def test_no_event_and_user_input_paths(self):
        self.check('no_event',['sync:ROCm0','async_try','sync:CUDA0','sync:ROCm0','buffer_try','get','set','graph'],
                   ['copy.scheduler_fallback'],['event.synchronize'])
        self.check('user_input',['event_sync','buffer_try','get','set','graph','event_record'],
                   ['copy.host_stage'],['copy.scheduler_async_attempt','copy.scheduler_fallback'])

    def test_buffer_direct_host_and_same_tensor_paths(self):
        self.check('buffer_accepted',['buffer_try'],['copy.buffer_interface_attempt'],['copy.host_stage'])
        self.check('host_source',['set'],['copy.host_to_backend'],['copy.host_stage'])
        self.check('host_destination',['get'],['copy.backend_to_host'],['copy.host_stage'])
        self.check('same_tensor',[],[],['copy.host_stage'])

    def test_public_async_fallback_and_accepted_copy_data(self):
        self.check('public_fallback',['async_try','sync:CUDA0','sync:ROCm0','buffer_try','get','set'],
                   ['copy.public_async_fallback','copy.host_stage'])
        self.check('public_accepted',['async_try'],[],['copy.public_async_fallback','copy.host_stage'])
