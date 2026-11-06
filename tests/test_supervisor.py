from helpers import thermal_fixture
"""CPU-only: synthetic processes/counters, no model or GPU backend loads."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from heterollm.hardware import clean_environment
from heterollm.pilot import fast_sample, maps_sample, validate
from heterollm.supervisor import Limits, live_group, supervise
from test_loading_mode import REAL_LINE
from test_evidence_sizes import rename_during_stat


def valid_output():
    text = 'load_mode = none\nn_ctx = 512\nCUDA0 KV buffer size = 1 MiB\n'
    text += '\n'.join(f'load_tensors: layer {i} assigned to device {"CUDA0" if i<41 else "ROCm0"}' for i in range(81))
    for d in ('CUDA0','ROCm0'):
        text += f'\n{d} model buffer size = 20000 MiB\nnode # 1 (MUL_MAT): op [{d} buffer] c=1: result\n'
    text += '\nusing device CUDA0 (NVIDIA GeForce RTX 5090) (0000:05:00.0)\nusing device ROCm0 (AMD Radeon AI PRO R9700) (0000:03:00.0)\n'
    return text


def gpu_fixture():
    return {d:{'temperature_c':30,'used_bytes':1024,'total_bytes':32*1024**3} for d in ('CUDA0','ROCm0')}


class SupervisorTests(unittest.TestCase):
    def run_case(self, code='print("42")', fault=None, limits=None, pre=None, validate_fn=None):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/'attempt'
            ready = Path(temp)/'ready'
            child = ('import os,signal,time,subprocess,sys; from pathlib import Path; '+
                     'signal.signal(signal.SIGTERM, signal.SIG_IGN); '+
                     f'Path({str(ready)!r}).write_text("ready"); '+code)
            def samplers(pid):
                count = 0
                def fast():
                    nonlocal count
                    count += 1
                    sample = fast_sample(pid)
                    # Safety stimuli are synthetic, independent of the CI host's RAM/swap.
                    sample['memory']['MemAvailable'] = 64*1024**3
                    sample['memory']['HardwareCorrupted'] = 0
                    sample['swap_used'] = 0
                    if ready.exists():
                        if fault=='ram':sample['memory']['MemAvailable']=1
                        if fault=='swap':sample['swap_used']=257*1024**2
                        if fault=='rss':sample['process']['VmHWM']=9*1024**3
                        if fault=='identity':sample['process']['start_ticks']+=count
                        if fault=='fast_error':raise OSError('synthetic proc failure')
                        if fault=='fast_hang':time.sleep(10)
                        if fault=='fast_dies':os._exit(3)
                        if fault=='oom':sample['oom_kill']+=10**6
                    return sample
                def gpu():
                    if fault=='gpu_error':raise OSError('synthetic missing sensor')
                    if fault=='gpu_hang':time.sleep(10)
                    data=gpu_fixture()
                    if ready.exists() and fault=='temperature':data['CUDA0']['temperature_c']=85
                    return data
                def maps():
                    if fault=='maps_hang':time.sleep(10)
                    data=maps_sample(pid, '/synthetic-model-no-file')
                    if ready.exists() and fault=='mmap':data['model_mappings']=['synthetic mapping']
                    return data
                thermal_count = 0
                def thermal():
                    nonlocal thermal_count
                    thermal_count += 1
                    if fault=='thermal_admission':return thermal_fixture(cpu=90)
                    if fault=='thermal_missing_admission':raise OSError('missing DIMM sensor')
                    r=thermal_fixture()
                    if fault=='dimm_warning':
                        r['sensors']['dimm-0051'].update(temperature_c=55,alarms={'temp1_max_alarm':1})
                    if fault=='dimm_sticky':
                        r['sensors']['dimm-0051']['alarms']={'temp1_max_alarm':1}
                    if fault in ('dimm_cool','dimm_never_cool','dimm_cool_missing','dimm_cool_critical'):
                        r['sensors']['dimm-0053']['temperature_c']=60
                        if thermal_count>=2:
                            if fault=='dimm_cool':r['sensors']['dimm-0053']['temperature_c']=59
                            if fault=='dimm_cool_missing':raise OSError('DIMM missing while cooling')
                            if fault=='dimm_cool_critical':r['sensors']['dimm-0053']['temperature_c']=65
                    if ready.exists():
                        if fault=='thermal_critical':r['sensors']['cpu']['temperature_c']=96
                        if fault=='thermal_sustained':r['sensors']['dimm-0053']['temperature_c']=60
                        if fault=='thermal_missing':raise OSError('physical DIMM disconnected')
                        if fault=='thermal_stale':r['started_monotonic']-=10
                        if fault=='thermal_hang':time.sleep(10)
                        if fault=='thermal_alarm':r['sensors']['dimm-0051']['alarms']={'temp1_crit_alarm':1}
                    return r
                return {'thermal':thermal,'fast':fast,'gpu':gpu,'maps':maps}
            r=supervise([sys.executable,'-c',child] if fault!='exec' else ['/no/such/executable'],directory,
                clean_environment(), pre or (lambda:{'swap_baseline':0}),samplers,
                validate_fn or (lambda out,err:{'errors':[] if out.strip()=='42' else ['wrong answer']}),
                limits or replace(Limits(),overall=5,preflight=2,term_grace=.05))
            saved={p.name:p.read_text() for p in directory.iterdir() if p.is_file()}
            self.assertEqual(json.loads(saved['result.json']),r)
            self.assertTrue(r['cleanup_complete'],r)
            self.assertTrue(r['reader_cleanup_complete'],r)
            if r['pid']:
                self.assertFalse(live_group(r['pid']))
                self.assertIn('wait4-rusage.json',saved)
            self.assertIn('"event": "final"',saved['events.jsonl'])
            self.assertLessEqual(r['launches'],1)
            return r,saved

    def test_host_thermal_admission_never_executes_without_two_valid_reads(self):
        for fault in ('thermal_admission','thermal_missing_admission'):
            r,s=self.run_case(fault=fault)
            self.assertEqual(r['launches'],0)
            self.assertNotIn('exec_release',s['events.jsonl'])

    def test_host_thermal_stops_preserve_telemetry_and_kill_owned_child(self):
        for fault,reason in [('thermal_critical','host_thermal_critical:cpu'),
                             ('thermal_alarm','host_thermal_hardware_alarm:dimm-0051'),
                             ('thermal_missing','monitor_or_supervisor_failure'),
                             ('thermal_stale','monitor_or_supervisor_failure'),
                             ('thermal_hang','monitor_or_supervisor_failure')]:
            r,s=self.run_case('time.sleep(60)',fault)
            self.assertEqual(r['reason'],reason,r)
            self.assertEqual(r['exit_code'],-signal.SIGKILL,r)
            self.assertIn('thermal.jsonl',s);self.assertIn('stop-intent.json',s)
            self.assertGreaterEqual(r['host_thermal']['valid_samples'],2)

    def test_sustained_host_thermal_stop_uses_existing_cleanup(self):
        # Deterministic elapsed-time policy is covered at its real 25 s above;
        # shorten only the CPU fixture's dwell, never production safety settings.
        with patch('heterollm.host_thermal.SUSTAINED_SECONDS',.5):
            r,s=self.run_case('time.sleep(60)','thermal_sustained')
        self.assertEqual(r['reason'],'host_thermal_sustained:dimm-0053',r)
        self.assertEqual(r['host_thermal']['max_observed_c']['dimm-0053'],60)
        self.assertIn('host_thermal_state',s['events.jsonl'])

    def test_dimm_warning_sticky_alarm_and_recovered_admission_execute_normally(self):
        for fault in ('dimm_warning','dimm_sticky','dimm_cool'):
            r,s=self.run_case('print("42",flush=True);time.sleep(.1)',fault)
            self.assertEqual(r['status'],'passed',r);self.assertEqual(r['launches'],1)
            self.assertIn('host_thermal_ready',s['events.jsonl'])
            if fault=='dimm_cool':
                self.assertIn('dimm_admission_cooling',s['events.jsonl'])
                self.assertGreaterEqual(r['host_thermal']['valid_samples'],3)
            else:self.assertGreater(r['host_thermal']['high_alarm_warnings']['dimm-0051']['asserted_samples'],0)

    def test_dimm_cooling_timeout_overall_deadline_missing_and_critical_cleanup_before_exec(self):
        with patch('heterollm.host_thermal.DIMM_ADMISSION_COOLDOWN_SECONDS',.5):
            r,s=self.run_case(fault='dimm_never_cool')
        self.assertEqual(r['reason'],'host_thermal_admission_cooldown_timeout',r)
        self.assertEqual(r['launches'],0)
        for fault,reason in [('dimm_never_cool','overall_timeout'),
                             ('dimm_cool_missing','monitor_or_supervisor_failure'),
                             ('dimm_cool_critical','host_thermal_critical:dimm-0053')]:
            limits=replace(Limits(),overall=1.5,preflight=1,term_grace=.05) if fault=='dimm_never_cool' else None
            r,s=self.run_case(fault=fault,limits=limits)
            self.assertEqual(r['reason'],reason,r);self.assertEqual(r['launches'],0)
            self.assertNotIn('exec_release',s['events.jsonl']);self.assertIn('stop-intent.json',s)

    def test_normal_completion_preserves_raw_and_rusage(self):
        r,s=self.run_case('print("42",flush=True);time.sleep(.3)')
        self.assertEqual(r['status'],'passed',r)
        self.assertEqual(s['stdout.txt'],'42\n')
        self.assertGreater(json.loads(s['wait4-rusage.json'])['rusage']['ru_maxrss'],0)
        self.assertIn('minor_faults',s['fast.jsonl'])
        self.assertIn('Active(anon)',s['fast.jsonl'])

    def test_wait4_peak_above_cap_rejects_spike_between_samples(self):
        real_wait4 = os.wait4
        def high_peak(*args):
            pid, status, usage = real_wait4(*args)
            if pid:
                values = list(usage)
                values[2] = 33*1024**2  # ru_maxrss is KiB on Linux
                usage = type(usage)(values)
            return pid, status, usage
        with patch('heterollm.supervisor.os.wait4',side_effect=high_peak):
            r,s=self.run_case('print("42",flush=True);time.sleep(.05)',
                             limits=replace(Limits(),rss_cap=32*1024**3))
        self.assertEqual(r['reason'],'process_rss_limit',r)
        self.assertEqual(r['peak_child_rss_bytes'],33*1024**3)
        self.assertIn('post_exit_rss_limit',s['events.jsonl'])

    def test_memory_swap_rss_thermal_and_mapping_stops(self):
        for fault,reason in [('ram','host_memory_pressure'),('swap','swap_growth'),('rss','process_rss_limit'),
                             ('temperature','temperature_limit'),('mmap','unexpected_model_mapping')]:
            with self.subTest(fault=fault):
                r,s=self.run_case('time.sleep(60)',fault)
                self.assertEqual(r['reason'],reason,r)
                self.assertEqual(r['exit_code'],-signal.SIGKILL,r)
                self.assertIn('stop-intent.json',s)
                self.assertTrue(json.loads(s['stop-intent.json'])['latest'])

    def test_bounded_timeout_includes_startup(self):
        r,s=self.run_case('time.sleep(60)',limits=replace(Limits(),overall=.15,preflight=1,term_grace=.05))
        self.assertEqual(r['reason'],'overall_timeout',r)
        self.assertLess(r['elapsed_seconds'],2)

    def test_monitor_errors_and_hangs(self):
        for fault in ('fast_error','fast_hang','fast_dies','gpu_error','gpu_hang','maps_hang','identity','oom'):
            with self.subTest(fault=fault):
                r,s=self.run_case('time.sleep(60)',fault)
                self.assertEqual(r['reason'],'monitor_or_supervisor_failure',r)
                self.assertIn('error',r)

    def test_preflight_failure_never_launches(self):
        def fail():raise ValueError('bad receipt')
        r,s=self.run_case(pre=fail)
        self.assertIsNone(r['pid']);self.assertEqual(r['launches'],0)
        self.assertNotIn('stdout.txt',s)

    def test_partial_monitor_launch_failure_cleans_up(self):
        from heterollm.supervisor import Reader
        def constructor(name,*args,**kwargs):
            if name=='gpu':raise OSError('synthetic worker creation failure')
            return Reader(name,*args,**kwargs)
        with patch('heterollm.supervisor.Reader',side_effect=constructor):
            r,s=self.run_case('time.sleep(60)')
        self.assertEqual(r['launches'],0)
        self.assertIn('synthetic worker creation failure',r['error'])

    def test_preflight_timeout_never_launches(self):
        r,s=self.run_case(pre=lambda:time.sleep(10),limits=replace(Limits(),preflight=.1,overall=2))
        self.assertEqual(r['reason'],'preflight_timeout')
        self.assertEqual(r['launches'],0)

    def test_external_child_term_and_kill(self):
        for sig in (signal.SIGTERM,signal.SIGKILL):
            with self.subTest(sig=sig):
                r,s=self.run_case(f'signal.signal(signal.SIGTERM,signal.SIG_DFL);os.kill(os.getpid(),{int(sig)})')
                self.assertEqual(r['exit_code'],-sig,r)
                self.assertEqual(r['status'],'failed')

    def test_supervisor_term_preserves_final_record(self):
        r,s=self.run_case('os.kill(os.getppid(),signal.SIGTERM);time.sleep(60)')
        self.assertEqual(r['reason'],'supervisor_signal')
        self.assertIn('stop-intent.json',s)

    def test_descendant_cleanup_after_direct_child_exit(self):
        code='c=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);print(c.pid,flush=True)'
        r,s=self.run_case(code,validate_fn=lambda o,e:{'errors':[]})
        descendant=int(s['stdout.txt'])
        p=Path(f'/proc/{descendant}/stat')
        self.assertTrue(not p.exists() or p.read_text().rsplit(')',1)[1].split()[0]=='Z')

    def test_exec_failure_is_durable(self):
        r,s=self.run_case(fault='exec')
        self.assertEqual(r['exit_code'],127)
        self.assertIn('exec failure',s['stderr.txt'])

    def test_raw_log_and_total_evidence_caps(self):
        r,s=self.run_case('print("x"*100000,flush=True);time.sleep(60)',
                         limits=replace(Limits(),log_cap=1000,term_grace=.05))
        self.assertEqual(r['reason'],'log_limit')
        r,s=self.run_case('time.sleep(60)',limits=replace(Limits(),evidence_cap=100,term_grace=.05))
        self.assertEqual(r['reason'],'evidence_limit')

    def test_atomic_rename_during_accounting_does_not_stop_child(self):
        with patch('heterollm.supervisor.evidence_sizes',side_effect=rename_during_stat) as scan:
            r,s=self.run_case('print("42",flush=True);time.sleep(.3)')
        self.assertEqual(r['status'],'passed',r)
        self.assertGreater(scan.call_count,0)

    def test_size_stops_still_fire_during_atomic_rename(self):
        for limits,reason in [(replace(Limits(),log_cap=1000,term_grace=.05),'log_limit'),
                              (replace(Limits(),evidence_cap=100,term_grace=.05),'evidence_limit')]:
            with self.subTest(reason=reason),patch('heterollm.supervisor.evidence_sizes',side_effect=rename_during_stat):
                r,_=self.run_case('print("x"*100000,flush=True);time.sleep(60)',limits=limits)
            self.assertEqual(r['reason'],reason,r)

    def test_unexpected_accounting_error_keeps_traceback_and_stops(self):
        def missing_permanent(directory):
            raise FileNotFoundError('synthetic missing permanent evidence')
        with patch('heterollm.supervisor.evidence_sizes',side_effect=missing_permanent):
            r,s=self.run_case('time.sleep(60)')
        self.assertEqual(r['reason'],'monitor_or_supervisor_failure')
        self.assertIn('missing_permanent',r['traceback'])
        self.assertIn('FileNotFoundError',r['traceback'])
        self.assertEqual(json.loads(s['stop-intent.json'])['traceback'],r['traceback'])

    def test_reader_exception_retains_origin_traceback(self):
        r,s=self.run_case(fault='gpu_error')
        self.assertEqual(r['reason'],'monitor_or_supervisor_failure')
        worker=json.loads(s['gpu-latest.json'])
        self.assertIn('synthetic missing sensor',worker['traceback'])
        self.assertIn('in gpu',worker['traceback'])
        self.assertIn('in poll',r['traceback'])

    def test_finalization_exception_retains_traceback_and_cleanup(self):
        def broken_validation(out,err):raise ValueError('synthetic validation crash')
        r,_=self.run_case(validate_fn=broken_validation)
        self.assertEqual(r['reason'],'finalization_failure')
        self.assertIn('broken_validation',r['finalization_traceback'])

    def test_fault_log_stops(self):
        r,s=self.run_case('print("HIP error: illegal memory access",file=sys.stderr,flush=True);time.sleep(60)')
        self.assertEqual(r['reason'],'gpu_failure')

    def test_wrong_loading_mode_stops(self):
        r,s=self.run_case('print("load_mode = mmap",file=sys.stderr,flush=True);time.sleep(60)')
        self.assertEqual(r['reason'],'unexpected_loading_mode')

    def test_real_loading_line_split_during_child_execution(self):
        split=REAL_LINE.index(b'none')+2
        code=f'os.write(2,{REAL_LINE[:split]!r});time.sleep(.15);os.write(2,{REAL_LINE[split:]!r});print("42",flush=True);time.sleep(.05)'
        r,s=self.run_case(code)
        self.assertEqual(r['status'],'passed',r)
        self.assertEqual(s['stderr.txt'].encode(),REAL_LINE)

    def test_complete_bad_mode_and_malformed_eof_stop(self):
        for line in (REAL_LINE.replace(b'none)',b'dio)'),b'(load_mode = none'):
            with self.subTest(line=line):
                r,s=self.run_case(f'os.write(2,{line!r});print("42",flush=True)')
                self.assertEqual(r['reason'],'unexpected_loading_mode',r)

    def test_other_guards_remain_active_during_incomplete_loader_line(self):
        for fault,reason in [('swap','swap_growth'),('ram','host_memory_pressure'),
                             ('rss','process_rss_limit'),('temperature','temperature_limit')]:
            with self.subTest(fault=fault):
                r,s=self.run_case('os.write(2,b"(load_mode = n");time.sleep(60)',fault)
                self.assertEqual(r['reason'],reason,r)

    def test_validation_failure_never_retries(self):
        r,s=self.run_case('print("42 with extra words")')
        self.assertEqual(r['reason'],'validation_failure');self.assertEqual(r['launches'],1)

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                supervise([],d,{},lambda:None,lambda pid:None,lambda o,e:None)

    def test_strict_real_evidence_acceptance(self):
        self.assertEqual(validate('42\n',valid_output())['errors'],[])
        self.assertTrue(validate('42\nExplanation',valid_output())['errors'])
        for missing in ('load_mode = none','CUDA0 model buffer size = 20000 MiB','[ROCm0 buffer] c=1:'):
            self.assertTrue(validate('42',valid_output().replace(missing,''))['errors'])

    def test_supervisor_kill_leaves_journal_and_child_death_signal(self):
        # Outer test owns the supervisor fixture; no real GPU tools involved.
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'attempt';ready=Path(d)/'ready'
            script=f'''
from tests_support import *
from pathlib import Path
child = "import os,time;from pathlib import Path;Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(60)"
supervise([sys.executable,'-c',child],{str(out)!r},clean_environment(),lambda:{{'swap_baseline':0}},
 lambda pid:{{'thermal':thermal_fixture,'fast':lambda:fast_sample(pid),'gpu':gpu_fixture,'maps':lambda:maps_sample(pid,'/synthetic')}},lambda o,e:{{'errors':[]}})
'''
            module=Path(d)/'tests_support.py'
            module.write_text('from helpers import thermal_fixture\nimport sys\nfrom heterollm.supervisor import supervise\nfrom heterollm.hardware import clean_environment\nfrom heterollm.pilot import fast_sample as real_fast,maps_sample\ndef fast_sample(pid):\n s=real_fast(pid);s["memory"]["MemAvailable"]=64*1024**3;s["memory"]["HardwareCorrupted"]=0;s["swap_used"]=0;return s\ndef gpu_fixture():\n return {"CUDA0":{"temperature_c":30},"ROCm0":{"temperature_c":30}}\n')
            env={**os.environ,'PYTHONPATH':str(Path.cwd())+os.pathsep+str(Path.cwd()/'tests')+os.pathsep+d}
            p=subprocess.Popen([sys.executable,'-c',script],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                end=time.monotonic()+10
                while not ready.exists() and time.monotonic()<end and p.poll() is None:time.sleep(.02)
                self.assertTrue(ready.exists(),'fixture failed readiness')
                pid=int(ready.read_text());p.kill();p.communicate(timeout=3)
                end=time.monotonic()+3
                while live_group(pid) and time.monotonic()<end:time.sleep(.02)
                self.assertFalse(live_group(pid))
                self.assertIn('exec_release',(out/'events.jsonl').read_text())
                self.assertEqual(json.loads((out/'result.json').read_text())['status'],'incomplete')
            finally:
                if p.poll() is None:p.kill()
                p.communicate(timeout=3)
