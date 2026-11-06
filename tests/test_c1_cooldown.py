from helpers import c1_cpu_plan
"""CPU-only thermal scheduling: synthetic temperatures and isolated readers."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from helpers import ROOT, config
from heterollm import stage_c as c0, stage_c1 as c1
from heterollm.gpu_sensors import SensorUnavailable
from heterollm.supervisor import live_group
from test_evidence_sizes import rename_during_stat
import test_stage_c1 as c1_tests

GIB = 1024**3


def counters():
    return {'memory':{'MemAvailable':64*GIB,'HardwareCorrupted':0},
            'swap_used':0,'oom_kill':0,
            'process':{'VmRSS':32*1024**2,'VmHWM':32*1024**2,'start_ticks':123}}


def sensors(temp=40):
    return {d:{'pci':pci,'temperature_c':temp,'total_bytes':32*GIB,'used_bytes':GIB}
            for d,pci in config().devices.items()}


class CooldownTests(unittest.TestCase):
    def simulate(self, gpu=None, fast=None, *, deadline=5000, fail_poll=None,
                 stale=None, close_failure=False, interrupt_at=None):
        clock = SimpleNamespace(now=0.)
        readers = []
        class FakeReader:
            def __init__(self,name,fn,directory,period,budget):
                self.name=name;self.period=period;self.next=0;self.seq=0
                self.process=SimpleNamespace(pid=1000+len(readers));self.closed=False
                readers.append(self)
            def poll(self):
                if fail_poll and self.name==fail_poll[0]:
                    raise RuntimeError(fail_poll[1])
                if clock.now < self.next or (stale==self.name and self.seq):
                    return None
                self.next=clock.now+self.period;self.seq+=1
                data=(gpu or (lambda t:sensors()))(clock.now) if self.name=='gpu' else (fast or (lambda t:counters()))(clock.now)
                return {'seq':self.seq,'started':{'monotonic':clock.now},
                        'ended':{'monotonic':clock.now},'data':data,'error':None}
            def close(self):
                self.closed=True
                return not (close_failure and self.name=='gpu')
        def sleep(seconds):clock.now+=max(seconds,.1)
        def check():
            if interrupt_at is not None and clock.now>=interrupt_at:
                raise InterruptedError('suite_signal')
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            d=Path(temp)/'cooldown'
            stack.enter_context(patch.object(c1,'Reader',FakeReader))
            stack.enter_context(patch.object(c1.time,'monotonic',side_effect=lambda:clock.now))
            stack.enter_context(patch.object(c1.time,'sleep',side_effect=sleep))
            value=c1.cooldown(config(output=str(d)),deadline,0,check)
            self.assertEqual(json.loads((d/'result.json').read_text()),value)
            self.assertTrue(all(r.closed for r in readers))
            return value,readers

    def test_already_cool_still_waits_at_least_thirty_seconds(self):
        value,_=self.simulate()
        self.assertEqual(value['status'],'ready')
        self.assertGreaterEqual(value['elapsed_seconds'],30)
        self.assertLess(value['elapsed_seconds'],31)
        self.assertGreaterEqual(value['stable_samples'],11)
        self.assertGreaterEqual(value['stable_seconds'],10)

    def test_saved_fifty_degree_failure_can_cool_within_bound(self):
        saved=json.loads((ROOT/'runs/llama33-70b/stage-c1-timed-20261020/03-r1-mixed-equal/gpu-readiness.json').read_text())
        self.assertEqual(saved['ROCm0']['temperature_c'],50)
        def cooling(t):
            data=deepcopy(saved)
            if t>=60:data['ROCm0']['temperature_c']=45
            return data
        value,_=self.simulate(gpu=cooling)
        self.assertEqual(value['status'],'ready')
        self.assertGreaterEqual(value['elapsed_seconds'],70)
        self.assertLess(value['elapsed_seconds'],75)

    def test_either_gpu_above_threshold_resets_stable_window(self):
        def rebound(t):
            data=sensors(45)
            if 25<=t<26:data['CUDA0']['temperature_c']=46
            return data
        value,_=self.simulate(gpu=rebound)
        self.assertEqual(value['status'],'ready')
        self.assertGreaterEqual(value['elapsed_seconds'],36)

    def test_never_cooling_times_out_at_ten_minutes(self):
        value,_=self.simulate(gpu=lambda t:sensors(50))
        self.assertEqual(value['status'],'failed')
        self.assertIn('temperature readiness deadline',value['error'])
        self.assertGreaterEqual(value['elapsed_seconds'],600)
        self.assertLess(value['elapsed_seconds'],601)

    def test_missing_invalid_or_wrong_identity_sensor_fails_closed(self):
        mutations=[lambda d:d.pop('ROCm0'),
                   lambda d:d['CUDA0'].pop('temperature_c'),
                   lambda d:d['ROCm0'].update(temperature_c=None),
                   lambda d:d['ROCm0'].update(pci='0000:99:00.0'),
                   lambda d:d['ROCm0'].update(used_bytes=33*GIB)]
        for mutation in mutations:
            def gpu(t):
                data=sensors();mutation(data);return data
            with self.subTest(mutation=mutation):
                value,_=self.simulate(gpu=gpu)
                self.assertEqual(value['status'],'failed')
                self.assertLess(value['elapsed_seconds'],1)

    def test_sensor_failure_timeout_and_stale_samples_stop(self):
        for failure in [('gpu','required GPU sensor unavailable'),('gpu','gpu: read timeout'),
                        ('fast','fast: startup timeout')]:
            value,_=self.simulate(fail_poll=failure)
            self.assertIn(failure[1],value['error'])
        for name in ('gpu','fast'):
            value,_=self.simulate(stale=name)
            self.assertEqual(value['status'],'failed')
            self.assertIn('stale sample',value['error'])

    def test_suite_budget_reserves_full_attempt_and_is_not_extended(self):
        value,readers=self.simulate(deadline=899)
        self.assertIn('suite time',value['error']);self.assertEqual(readers,[])
        value,_=self.simulate(deadline=930)
        self.assertEqual(value['status'],'failed')
        self.assertIn('suite time',value['error'])
        self.assertLess(value['elapsed_seconds'],31)
        value,_=self.simulate(deadline=1000,gpu=lambda t:sensors(50))
        self.assertIn('suite time',value['error'])
        self.assertLess(value['elapsed_seconds'],101)

    def test_runtime_temperature_memory_swap_fault_and_oom_stops_remain(self):
        value,_=self.simulate(gpu=lambda t:sensors(85))
        self.assertIn('temperature_limit',value['error'])
        cases=[('host_memory_pressure',lambda d:d['memory'].update(MemAvailable=3*GIB)),
               ('swap_growth',lambda d:d.update(swap_used=257*1024**2)),
               ('hardware_memory_error',lambda d:d['memory'].update(HardwareCorrupted=4096)),
               ('process_rss_limit',lambda d:d['process'].update(VmHWM=33*GIB)),
               ('OOM',lambda d:d.update(oom_kill=1))]
        for reason,mutate in cases:
            def fast(t):
                d=counters()
                if t>1:mutate(d)
                return d
            value,_=self.simulate(fast=fast)
            self.assertEqual(value['status'],'failed')
            self.assertIn(reason,value['error'])

    def test_signal_and_reader_cleanup_failure_prevent_ready(self):
        value,_=self.simulate(interrupt_at=2)
        self.assertIn('suite_signal',value['error'])
        value,readers=self.simulate(close_failure=True)
        self.assertEqual(value['status'],'failed')
        self.assertFalse(value['reader_cleanup_complete'])
        self.assertTrue(all(r.closed for r in readers))

    def test_atomic_rename_during_cooldown_accounting_succeeds(self):
        with patch.object(c1,'evidence_sizes',side_effect=rename_during_stat) as scan:
            value,_=self.simulate()
        self.assertEqual(value['status'],'ready',value)
        self.assertGreater(scan.call_count,0)

    def test_actual_evidence_over_cap_still_stops_during_rename(self):
        def oversized(directory):
            (Path(directory)/'fast.jsonl').write_bytes(b'x'*8192)
            return rename_during_stat(directory)
        with patch.object(c1,'LIMITS',replace(c1.LIMITS,evidence_cap=4096)), \
             patch.object(c1,'evidence_sizes',side_effect=oversized):
            value,_=self.simulate()
        self.assertEqual(value['status'],'failed')
        self.assertIn('evidence_limit',value['error'])
        self.assertTrue(value['reader_cleanup_complete'])

    def test_unexpected_accounting_failure_preserves_traceback(self):
        def missing_permanent(directory):
            raise FileNotFoundError('synthetic missing permanent evidence')
        with patch.object(c1,'evidence_sizes',side_effect=missing_permanent):
            value,_=self.simulate()
        self.assertEqual(value['status'],'failed')
        self.assertIn('missing_permanent',value['traceback'])
        self.assertIn('FileNotFoundError',value['traceback'])
        self.assertTrue(value['reader_cleanup_complete'])

    def real_reader_case(self,gpu):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            d=Path(temp)/'cooldown'
            stack.enter_context(patch.object(c1,'sampler_factory',return_value={
                'fast':counters,'gpu':gpu}))
            stack.enter_context(patch.object(c0,'COOLDOWN_SECONDS',.05))
            stack.enter_context(patch.object(c1,'COOLDOWN_STABLE_SECONDS',.01))
            stack.enter_context(patch.object(c1,'COOLDOWN_STABLE_SAMPLES',2))
            value=c1.cooldown(config(output=str(d)),time.monotonic()+1000,0,lambda:None)
            self.assertTrue(value['reader_cleanup_complete'])
            for r in value['reader_cleanup']:
                self.assertEqual(live_group(r['pid']),[])
            logs={p.name:p.read_text() for p in d.glob('*.json*')}
            return value,logs

    def test_real_readers_cool_successfully_and_are_reaped(self):
        value,logs=self.real_reader_case(sensors)
        self.assertEqual(value['status'],'ready',value)
        self.assertIn('gpu.jsonl',logs);self.assertIn('fast.jsonl',logs)

    def test_real_sensor_failure_keeps_raw_evidence_and_cleans_up(self):
        def broken():
            raise SensorUnavailable('ROCm0',{'raw_temperature':None,'pci':'0000:03:00.0'},'missing sensor')
        value,logs=self.real_reader_case(broken)
        self.assertEqual(value['status'],'failed')
        evidence=json.loads(logs['gpu-latest.json'])
        self.assertIsNone(evidence['data']['raw_temperature'])
        self.assertIn('missing sensor',evidence['error'])

    def test_hung_reader_is_bounded_and_cleaned_up(self):
        def hung():time.sleep(20)
        start=time.monotonic();value,_=self.real_reader_case(hung)
        self.assertEqual(value['status'],'failed')
        self.assertIn('timeout',value['error'])
        self.assertLess(time.monotonic()-start,5)


class ThermalSuiteTests(unittest.TestCase):
    def test_manifest_protocol_and_model_commands_are_unchanged(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        conf,plan=c1.load_plan(c1_cpu_plan(temp.name))
        old=json.loads((ROOT/'configs/stage-c1-manifest.json').read_text())
        new=c1.manifest(conf,plan)
        self.assertEqual(new['cooldown_protocol'],{
            'before_every_launch':True,'minimum_seconds':30,'maximum_seconds':600,
            'stable_seconds':10,'minimum_consecutive_gpu_samples':11,
            'temperature_ceiling_c':45,'gpu_sample_period_seconds':1,
            'next_attempt_reserved_seconds':900})
        self.assertEqual(new['limits'],old['limits'])
        self.assertEqual(new['suite_seconds'],old['suite_seconds'])
        self.assertEqual(new['placement_contract'],old['placement_contract'])
        old_by_round_id={(a['round'],a['id']):a for a in old['launches']}
        for b in new['launches']:
            a=old_by_round_id[b['round'],b['id']]
            for key in ('id','round','argv','environment','policy'):self.assertEqual(a[key],b[key])
            self.assertNotEqual(a['output'],b['output'])
        with self.assertRaisesRegex(ValueError,'manifest/source'):
            c1.load_plan(ROOT/'configs/stage-c1-plan.json')
        with self.assertRaisesRegex(ValueError,'manifest/source'):
            c1.load_plan(ROOT/'configs/stage-c1-thermal-plan.json')

    def test_cooldown_failure_stops_before_next_launch_and_persists(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            for n in ('hardware','runtime'):(d/(n+'.json')).write_text('{}')
            plan={'reference_sha256':'fixture','hardware_baseline':str(d/'hardware.json'),
                  'runtime_baseline':str(d/'runtime.json')}
            conf=config(output=str(d/'suite'));seen=[]
            def gate(cfg,deadline,baseline,check):
                seen.append((cfg.output,deadline,baseline))
                return {'status':'ready' if len(seen)<3 else 'failed','error':'cooldown timeout',
                        'reader_cleanup_complete':True}
            records=c1_tests.C1ContractTests().records()
            calls=[]
            def model(*args,**kwargs):
                calls.append(args)
                Path(args[1]).mkdir()
                return {'status':'passed','launches':1,'validation':{'metrics':records[len(calls)-1]['metrics']}}
            with patch.object(c1,'cooldown',side_effect=gate),patch.object(c1,'supervise',side_effect=model):
                result=c1.execute(conf,plan)
            self.assertEqual(result['launches'],2);self.assertEqual(len(calls),2)
            self.assertEqual(len(result['cooldowns']),3);self.assertEqual(result['status'],'failed')
            self.assertIsNone(result['comparison'])
            self.assertEqual(len({x[1] for x in seen}),1)
            self.assertEqual(len({x[2] for x in seen}),1)
            self.assertEqual(json.loads((Path(conf.output)/'suite-result.json').read_text()),result)

    def test_historical_partial_measurements_cannot_complete_new_comparison(self):
        old=json.loads((ROOT/'runs/llama33-70b/stage-c1-timed-20261020/suite-result.json').read_text())
        fresh=c1_tests.C1ContractTests().records()
        historical=old['measurements'][0]
        index=next(i for i,r in enumerate(fresh)
                   if r['round']==historical['round']
                   and r['policy']['name']==historical['policy']['name'])
        fresh[index]=historical
        with self.assertRaisesRegex(ValueError,'differ'):c1.summarize(fresh)
