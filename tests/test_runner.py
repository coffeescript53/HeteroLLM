from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from heterollm.cli import export_result
from heterollm.hardware import clean_environment,file_identity
from heterollm.runner import (Session,StopSuite,classify,execution_lock,load_receipt,run_process,
                              safety_reason,unique_directory,verify_model,write_json)
from helpers import ROOT,config,snapshot,tiny_gguf


class ProcessTests(unittest.TestCase):
    def test_success_and_raw_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run'
            r=run_process([sys.executable,'-c','print("raw");raise SystemExit(7)'],p,3,clean_environment())
            self.assertEqual(r['exit_code'],7)
            self.assertEqual((p/'stdout.txt').read_text(),'raw\n')
            self.assertEqual(r['classification'],'runtime_failure')
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';p.mkdir();(p/'stdout.txt').write_text('keep')
            with self.assertRaises(FileExistsError):run_process([sys.executable,'-c',''],p,1,clean_environment())
            self.assertEqual((p/'stdout.txt').read_text(),'keep')
            with self.assertRaises(FileExistsError):write_json(p/'stdout.txt',{})
    def test_unique_paths(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertNotEqual(unique_directory(d,'test'),unique_directory(d,'test'))
    def test_timeout_reaps_owned_child(self):
        self.check_timeout_after_child_ready(startup_delay=0)
    def test_timeout_reaps_child_after_slow_startup(self):
        # Deliberately exceeds the old 0.25-second startup assumption.
        self.check_timeout_after_child_ready(startup_delay=0.5)
    def check_timeout_after_child_ready(self,startup_delay):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run'
            ready=Path(d)/'child-ready'
            child=('import os,time; from pathlib import Path; '
                   f'time.sleep({startup_delay!r}); '
                   f'p=Path({str(ready)!r}); '
                   'p.with_suffix(".tmp").write_text(str(os.getpid())); '
                   'p.with_suffix(".tmp").replace(p); time.sleep(60)')
            code=('import subprocess,sys; '
                  f'c=subprocess.Popen([sys.executable,"-c",{child!r}]); c.wait()')
            startup_started=time.monotonic()
            ready_at=None
            startup_expired=False
            def after_readiness():
                nonlocal ready_at,startup_expired
                now=time.monotonic()
                if ready_at is None:
                    if now-startup_started >= 10:
                        startup_expired=True
                        return 0.25  # Trigger owned-group cleanup, then fail below.
                    if ready.exists():
                        ready_at=now
                    return 0.0
                return now-ready_at
            # Only this runner call sees the test clock. Subprocess cleanup and
            # the 10-second startup bound use real time. Production deadlines
            # remain unchanged and still include process startup.
            clock=SimpleNamespace(monotonic=after_readiness,sleep=time.sleep)
            with patch('heterollm.runner.time',clock):
                r=run_process([sys.executable,'-c',code],p,0.25,clean_environment())
            self.assertFalse(startup_expired,'child readiness handshake exceeded 10 seconds')
            self.assertIsNotNone(ready_at,'child never completed its readiness handshake')
            self.assertEqual(r['classification'],'timeout',r)
            self.assertGreaterEqual(r['elapsed_seconds'],0.25)
            pid=int(ready.read_text())
            status=Path(f'/proc/{pid}/stat')
            self.assertTrue(not status.exists() or status.read_text().split()[2]=='Z')
    def test_production_timeout_includes_startup(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run'
            r=run_process([sys.executable,'-c','import time;time.sleep(60)'],p,0.25,clean_environment())
            self.assertEqual(r['classification'],'timeout')
            self.assertGreaterEqual(r['elapsed_seconds'],0.25)
    def test_cleanup_child_after_parent_exit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run'
            code='import subprocess,sys; c=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]);print(c.pid,flush=True)'
            r=run_process([sys.executable,'-c',code],p,3,clean_environment())
            self.assertEqual(r['classification'],'ok')
            pid=int((p/'stdout.txt').read_text());time.sleep(.05)
            status=Path(f'/proc/{pid}/stat')
            self.assertTrue(not status.exists() or status.read_text().split()[2]=='Z')
    def test_missing_executable_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run'
            r=run_process(['/does-not-exist'],p,2,clean_environment())
            self.assertNotEqual(r['classification'],'ok')
            self.assertTrue((p/'process.json').exists())
    def test_monitor_failure_stops_child(self):
        with tempfile.TemporaryDirectory() as d:
            def failure():raise RuntimeError('fixture monitor failure')
            r=run_process([sys.executable,'-c','import time;time.sleep(60)'],Path(d)/'run',3,clean_environment(),failure)
            self.assertEqual(r['classification'],'monitor_failure')
    def test_resource_stop(self):
        with tempfile.TemporaryDirectory() as d:
            r=run_process([sys.executable,'-c','import time;time.sleep(60)'],Path(d)/'run',3,clean_environment(),lambda:({},'swap_growth'))
            self.assertEqual(r['classification'],'swap_growth')
    def test_log_limit_stops_owned_process(self):
        with tempfile.TemporaryDirectory() as d:
            r=run_process([sys.executable,'-c','import sys,time;sys.stdout.write("x"*2000000);sys.stdout.flush();time.sleep(60)'],
                          Path(d)/'run',3,clean_environment(),max_log_mib=1)
            self.assertEqual(r['classification'],'log_limit')
    def test_lock_prevents_parallel_suites(self):
        with execution_lock():
            with self.assertRaises(ValueError):
                with execution_lock():pass
    def test_failure_classes(self):
        self.assertEqual(classify(1,'out of memory'),'memory_failure')
        self.assertEqual(classify(1,'HIP error: illegal memory access'),'gpu_failure')
    def test_missing_telemetry_not_zero_or_fatal(self):
        s=snapshot()['telemetry'];s['nvidia'][0]['temperature_c']=None
        self.assertIsNone(safety_reason(s,config(),0))
    def test_safety_limits(self):
        s=snapshot()['telemetry'];s['memory']['SwapFree']-=300*1024**2
        self.assertEqual(safety_reason(s,config(),0),'swap_growth')
        s=snapshot()['telemetry'];s['nvidia'][0]['temperature_c']=90
        self.assertEqual(safety_reason(s,config(),0),'temperature_limit')
        s=snapshot()['telemetry'];s['memory']['MemAvailable']=1
        self.assertEqual(safety_reason(s,config(),0),'host_memory_pressure')


class IntegrationTests(unittest.TestCase):
    """All measurements in these tests are intentionally fabricated protocol fixtures."""
    def make_config(self,d):
        model=Path(d)/'tiny.gguf';tiny_gguf(model)
        ident=file_identity(model)
        fake=str(ROOT/'tests/fixtures/fake_runtime.py')
        return config(model=str(model),model_size_bytes=ident['size_bytes'],model_sha256=ident['sha256'],
                      bench=fake,cli=fake,output=str(Path(d)/'runs'),environment={})
    def test_receipt_checks_model_changes(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.make_config(d);receipt=Path(d)/'receipt.json'
            verify_model(c,receipt)
            self.assertEqual(load_receipt(c,receipt)['sha256'],c.model_sha256)
            os.utime(c.model,ns=(1,1))
            with self.assertRaises(ValueError):load_receipt(c,receipt)
    def test_digest_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            c=replace(self.make_config(d),model_sha256='0'*64)
            with self.assertRaises(ValueError):verify_model(c,Path(d)/'receipt.json')
    def test_complete_offline_search_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.make_config(d);receipt=Path(d)/'receipt.json';verify_model(c,receipt)
            with patch('heterollm.runner.discover',return_value=snapshot()),patch('heterollm.runner.telemetry',return_value=snapshot()['telemetry']):
                s=Session(c,receipt)
                result=s.tune()
            self.assertEqual(result['selected'],'cuda-only')
            self.assertEqual(result['strongest_baseline'],'cuda-only')
            self.assertEqual(s.count,30)
            self.assertEqual(result['groups']['cuda-only']['tg']['n'],3)
            exported=Path(export_result(s.directory/'selection.json',Path(d)/'export'))
            self.assertEqual(json.loads((exported/'policy.json').read_text())['policy']['name'],'cuda-only')
            self.assertTrue((exported/'report.md').exists())
    def test_no_benchmark_without_validation(self):
        from heterollm.config import Policy
        with tempfile.TemporaryDirectory() as d:
            c=self.make_config(d);receipt=Path(d)/'receipt.json';verify_model(c,receipt)
            with patch('heterollm.runner.discover',return_value=snapshot()):
                s=Session(c,receipt)
                with self.assertRaises(ValueError):s.benchmark(Policy('cuda-only',('CUDA0',),None,3),'final',3)
    def test_invalid_answer_preserves_raw_output_and_blocks_diagnostic(self):
        from heterollm.config import Policy
        with tempfile.TemporaryDirectory() as d:
            c=self.make_config(d);receipt=Path(d)/'receipt.json';verify_model(c,receipt)
            raw='42\nThis is incorrect\n'
            def invalid_response(argv,path,diagnostic=False):
                path.mkdir()
                (path/'stdout.txt').write_text(raw)
                (path/'stderr.txt').write_text('')
                return {'classification':'ok'}
            with patch('heterollm.runner.discover',return_value=snapshot()):
                s=Session(c,receipt)
                with patch.object(s,'launch',side_effect=invalid_response) as launch, \
                     patch('heterollm.runner.verify_placement',return_value=[]):
                    policy=Policy('cuda-only',('CUDA0',),None,3)
                    result=s.validate(policy)
                self.assertEqual(result['status'],'invalid')
                self.assertIn('natural-language arithmetic smoke test failed',result['errors'])
                self.assertEqual(launch.call_count,1)
                self.assertFalse((Path(result['directory'])/'diagnostic').exists())
                self.assertEqual((Path(result['directory'])/'functional/stdout.txt').read_text(),raw)
                with self.assertRaises(ValueError):s.benchmark(policy,'final',3)
    def test_launch_bound_before_process(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.make_config(d);receipt=Path(d)/'receipt.json';verify_model(c,receipt)
            with patch('heterollm.runner.discover',return_value=snapshot()):
                s=Session(c,receipt);s.count=c.max_launches
                with self.assertRaises(StopSuite):s.launch([sys.executable,'-c',''],Path(d)/'never')
            self.assertFalse((Path(d)/'never').exists())
    def test_characterized_cli_response_uses_stderr_before_diagnostic(self):
        from heterollm.config import Policy
        from test_assistant_response import STDOUT, STDERR
        with tempfile.TemporaryDirectory() as d:
            c=self.make_config(d);receipt=Path(d)/'receipt.json';verify_model(c,receipt)
            def saved_output(argv,path,diagnostic=False):
                path.mkdir()
                (path/'stdout.txt').write_text(STDOUT if path.name=='functional' else '[]')
                (path/'stderr.txt').write_text(STDERR)
                return {'classification':'ok'}
            with patch('heterollm.runner.discover',return_value=snapshot()):
                s=Session(c,receipt)
                with patch.object(s,'launch',side_effect=saved_output) as launch, \
                     patch('heterollm.runner.verify_placement',return_value=[]):
                    result=s.validate(Policy('mixed-default',('CUDA0','ROCm0'),None,999))
                self.assertEqual(launch.call_count,2)
                self.assertNotIn('natural-language arithmetic smoke test failed',result['errors'])
    def test_binary_revision_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.make_config(d);receipt=Path(d)/'receipt.json';verify_model(c,receipt)
            s=snapshot();s['versions']['bench']['stdout']='commit 12345678'
            with patch('heterollm.runner.discover',return_value=s),self.assertRaises(ValueError):
                Session(c,receipt)
