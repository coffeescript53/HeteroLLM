from helpers import thermal_fixture, c1_cpu_plan
"""C1 preparation: preserved C0 replay, synthetic counters and Python children only."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

from helpers import config, ROOT
from heterollm import stage_c as c0, stage_c1 as c1
from heterollm.config import digest
from heterollm.pilot import fast_sample, maps_sample
from heterollm.supervisor import supervise
from test_stage_c import rows
from test_supervisor import gpu_fixture

GIB=1024**3


def timed_rows(conf,policy):
    data=rows(conf,policy)
    for row in data:row['samples_ns']=[1_000_000_000,2_000_000_000,4_000_000_000]
    return data


class C1ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        fixtures=json.loads(gzip.decompress((ROOT/'tests/fixtures/c0_allocation_evidence.json.gz').read_bytes()))
        cls.paths={};cls.profiles={}
        for label,files in fixtures.items():
            directory=cls.root/label;directory.mkdir()
            for name,text in files.items():(directory/name).write_text(text)
            cls.paths[label]=directory
            cls.profiles[label]=c1.calibration(directory,c1.BY_ID[label])

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_saved_c0_resources_pass_all_four_allocation_checks(self):
        for label,policy in c1.BY_ID.items():
            with self.subTest(label=label):
                proof=c1.allocation_check(self.paths[label],policy,self.profiles[label])
                self.assertEqual(proof['status'],'allocation_consistent_with_C0')
                self.assertGreaterEqual(proof['matching_samples'],3)
                self.assertGreaterEqual(proof['matching_span_seconds'],2)
                self.assertIsNone(proof['fresh_observed_layer_map'])
                self.assertEqual(len(proof['c0_observed_layer_map']),81)

    def test_equal_and_default_are_independently_distinguishable(self):
        for label,other in [('E','D'),('D','E')]:
            with self.assertRaises(ValueError):
                c1.allocation_check(self.paths[label],c1.BY_ID[label],self.profiles[other])

    def check_mutation(self,label,mutate):
        base,active=c1.active_samples(self.paths[label]);mutate(base,active)
        with patch.object(c1,'active_samples',return_value=(base,active)):
            with self.assertRaises(ValueError):c1.allocation_check('unused',c1.BY_ID[label],self.profiles[label])

    def test_wrong_gpu_missing_allocation_background_memory_or_activity_fail(self):
        def wrong_gpu(base,active):
            for row in active['gpu']:row['data']['ROCm0']['used_bytes']=base['ROCm0']['used_bytes']
        def extra_memory(base,active):
            for row in active['gpu']:row['data']['CUDA0']['used_bytes']+=256*1024**2
        def missing_activity(base,active):
            for row in active['gpu']:row['data']['ROCm0']['busy_percent']=None
        def flat_activity(base,active):
            for row in active['gpu']:row['data']['CUDA0']['utilization_percent']=0
        for change in (wrong_gpu,extra_memory,missing_activity,flat_activity):
            self.check_mutation('E',change)

    def test_cpu_offload_rss_or_user_ticks_missing_fails(self):
        for label in ('A','B'):
            def low_rss(base,active):
                for row in active['fast']:row['data']['process']['VmHWM']=2*GIB
            def extra_rss(base,active):
                for row in active['fast']:row['data']['process']['VmHWM']+=2*GIB
            def no_cpu(base,active):
                for row in active['fast']:row['data']['process']['user_ticks']=0
            for change in (low_rss,extra_rss,no_cpu):self.check_mutation(label,change)

    def test_short_allocation_match_is_insufficient(self):
        def one_sample(base,active):active['gpu']=active['gpu'][-1:]
        self.check_mutation('E',one_sample)

    def test_incomplete_or_stale_or_errored_resource_logs_fail(self):
        label='E'
        for target,mutate in [
            ('events.jsonl',lambda rows:[r for r in rows if r['event']!='reaped']),
            ('events.jsonl',lambda rows:rows+[next(r for r in rows if r['event']=='exec_release')]),
            ('gpu.jsonl',lambda rows:[{**rows[0],'error':'missing sensor'},*rows[1:]]),
            ('gpu.jsonl',lambda rows:rows[1:]),
            ('fast.jsonl',lambda rows:rows[:1])]:
            with tempfile.TemporaryDirectory() as temp:
                d=Path(temp)
                for name in ('events.jsonl','gpu.jsonl','fast.jsonl'):
                    values=c1.read_records(self.paths[label]/name)
                    if name==target:values=mutate(values)
                    (d/name).write_text(''.join(json.dumps(v)+'\n' for v in values))
                with self.assertRaises(ValueError):c1.active_samples(d)

    def test_stale_preexec_sensor_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            for name in ('events.jsonl','gpu.jsonl','fast.jsonl'):
                values=c1.read_records(self.paths['E']/name)
                if name=='gpu.jsonl':
                    values[0]['ended']['monotonic']-=10
                    values[0]['started']['monotonic']-=10
                (d/name).write_text(''.join(json.dumps(v)+'\n' for v in values))
            with self.assertRaisesRegex(ValueError,'stale'):c1.active_samples(d)

    def test_default_preflight_split_change_stops_without_substitution(self):
        conf=config()
        pre=json.loads((self.paths['D']/'preflight.json').read_text())
        pre['screening']['estimated_layers']['CUDA0'].append(40)
        pre['screening']['estimated_layers']['ROCm0'].remove(40)
        with patch.object(c1,'verify_reference',return_value={'profiles':self.profiles}), \
             patch.object(c0,'preflight',return_value=pre):
            with self.assertRaisesRegex(ValueError,'predicted placement changed'):
                c1.preflight(conf,{'reference_sha256':'x'},'D',0)

    def test_matching_preflight_reuses_full_c0_admission(self):
        pre=json.loads((self.paths['A']/'preflight.json').read_text())
        with patch.object(c1,'verify_reference',return_value={'profiles':self.profiles}), \
             patch.object(c0,'preflight',return_value=pre) as original:
            value=c1.preflight(config(),{'reference_sha256':'x'},'A',123)
            original.assert_called_once_with(config(),{'reference_sha256':'x'},c1.BY_ID['A'],123)
            self.assertFalse(value['c0_reference']['fresh_per_layer_trace'])


class C1ContractTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.plan_path=c1_cpu_plan(temp.name)

    def test_exact_order_repetitions_quiet_environment_and_shared_limits(self):
        conf=config()
        m=c1.manifest(conf,{'reference_sha256':'x'})
        self.assertEqual([r['id'] for r in m['launches']],list('EDABDBEABADE'))
        self.assertEqual(len({r['output'] for r in m['launches']}),12)
        self.assertEqual(asdict(c1.LIMITS),asdict(c0.C0_LIMITS))
        for r in m['launches']:
            argv=r['argv'];p=c1.BY_ID[r['id']]
            expected=c0.command(conf,p);expected.remove('-v');expected[expected.index('-r')+1]='3'
            self.assertEqual(argv,expected)
            self.assertNotIn('-v',argv);self.assertNotIn('GGML_SCHED_DEBUG',r['environment'])
            self.assertNotIn('LD_PRELOAD',r['environment'])
            self.assertEqual(argv[argv.index('-lm')+1],'none')
            self.assertEqual(argv[argv.index('-ngl')+1],str(p.gpu_layers))
        self.assertEqual(m['suite_seconds'],11400)

    def test_preview_cannot_run_hardware_or_processes(self):
        with patch.object(c1,'execute',side_effect=AssertionError('execution')), \
             patch.object(c0,'discover',side_effect=AssertionError('hardware')),redirect_stdout(io.StringIO()) as out:
            self.assertEqual(c1.main(['--plan',str(self.plan_path)]),0)
        value=json.loads(out.getvalue());self.assertEqual(value['mode'],'preview_only')
        self.assertEqual(len(value['manifest']['launches']),12);self.assertFalse(value['output_exists'])

    def test_activated_order_matches_reviewed_draft_but_old_manifests_are_refused(self):
        conf,plan=c1.load_plan(self.plan_path)
        current=c1.manifest(conf,plan)
        draft=json.loads((ROOT/'docs/STAGE_C1_MIXED_FIRST_PROPOSED_MANIFEST.json').read_text())
        for a,b in zip(current['launches'],draft['launches']):
            for key in ('ordinal','round','id','policy','argv','environment'):
                self.assertEqual(a[key],b[key])
            self.assertNotEqual(a['output'],b['output'])
        for name in ('stage-c1-plan.json','stage-c1-thermal-plan.json',
                     'stage-c1-evidence-fixed-plan.json','stage-c1-mixed-first-plan.json',
                     'stage-c1-memtotal-fixed-plan.json'):
            with self.subTest(plan=name),self.assertRaisesRegex(ValueError,'manifest/source'):
                c1.load_plan(ROOT/'configs'/name)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'stale-draft-plan.json'
            path.write_text(json.dumps({**plan,
                'manifest':str(ROOT/'docs/STAGE_C1_MIXED_FIRST_PROPOSED_MANIFEST.json')}))
            with self.assertRaisesRegex(ValueError,'manifest/source'):
                c1.load_plan(path)

    def test_reference_manifest_and_source_hash_tampering_refused(self):
        conf,original=c1.load_plan(self.plan_path)
        for field,value in [('stage','C0'),('reference_sha256','0'*64)]:
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'plan.json';p.write_text(json.dumps({**original,field:value}))
                with self.assertRaises(ValueError):c1.load_plan(p)
        for change in ('order','argv','source'):
            with tempfile.TemporaryDirectory() as temp:
                d=Path(temp);m=c1.manifest(conf,original)
                if change=='order':m['launches'].reverse()
                elif change=='argv':m['launches'][0]['argv'].append('-v')
                else:m['source_sha256']['heterollm/stage_c1.py']='0'*64
                (d/'manifest.json').write_text(json.dumps(m))
                (d/'plan.json').write_text(json.dumps({**original,'manifest':str(d/'manifest.json')}))
                with self.assertRaisesRegex(ValueError,'manifest/source'):c1.load_plan(d/'plan.json')

    def test_execution_caps_and_workload_tampering_refused(self):
        _,plan=c1.load_plan(self.plan_path)
        original=json.loads(Path(plan['config']).read_text())
        for change in ({'max_launches':13},{'suite_seconds':14400},{'threads':8},{'max_temperature_c':90},
                       {'minimum_available_ram_mib':1024},{'max_swap_growth_mib':128}):
            with tempfile.TemporaryDirectory() as temp:
                d=Path(temp);(d/'config.json').write_text(json.dumps({**original,**change}))
                (d/'plan.json').write_text(json.dumps({**plan,'config':str(d/'config.json')}))
                with self.assertRaises(ValueError):c1.load_plan(d/'plan.json')

    def test_actual_reference_hashes_and_recomputed_profiles(self):
        _,plan=c1.load_plan(self.plan_path)
        ref=c1.verify_reference(plan)
        self.assertEqual(set(ref['profiles']),set('ABDE'))
        self.assertEqual(len(ref['files']),163)

    def test_rows_require_three_and_reject_c0_as_timed(self):
        conf=config();p=c1.BY_ID['E']
        with tempfile.TemporaryDirectory() as temp:
            conf=replace(conf,output=temp)
            (Path(temp)/'preflight.json').write_text('{"c0_reference":{"profile":{}}}')
            with patch.object(c1,'allocation_check',return_value={'status':'fixture'}):
                good=c1.validate(conf,p,json.dumps(timed_rows(conf,p)),'')
                self.assertEqual(good['errors'],[])
                self.assertEqual(good['metrics']['pp']['n'],3)
                for count in (0,1,2,4):
                    data=timed_rows(conf,p);data[0]['samples_ns']=[1]*count
                    self.assertTrue(c1.validate(conf,p,json.dumps(data),'')['errors'])
                for field,value in [('load_mode','mmap'),('lazy_mode','on'),('devices','ROCm0'),
                                    ('tensor_split','2/1'),('n_depth',128),('samples_ns',[1,2,-3])]:
                    data=timed_rows(conf,p);data[0][field]=value
                    self.assertTrue(c1.validate(conf,p,json.dumps(data),'')['errors'])
                for log in ('node # 1','load_tensors: layer 40 assigned to device CUDA0'):
                    self.assertTrue(c1.validate(conf,p,json.dumps(timed_rows(conf,p)),log)['errors'])

    def records(self):
        records=[]
        for number,order in enumerate(c1.ORDER,1):
            for label in order:
                policy=c1.BY_ID[label]
                metrics=c1.parse_benchmark(timed_rows(config(),policy),config(),3,load_mode='none')
                for m in metrics.values():m['mean']*=number
                records.append({'run_id':f'{number}-{label}','round':number,'policy':policy.to_dict(),
                                'status':'ok','phase':'c1','comparison_key':'same','validation_key':label,
                                'metrics':metrics})
        return records

    def test_statistics_have_three_process_observations_not_nine(self):
        result=c1.summarize(self.records())
        self.assertEqual(result['independent_n_per_configuration'],3)
        self.assertFalse(result['includes_C0'])
        for group in result['groups'].values():
            self.assertEqual(group['pp']['n'],3);self.assertEqual(group['tg']['n'],3)
            self.assertEqual(len(group['run_ids']),3)

    def test_incomplete_duplicate_round_or_c0_records_cannot_form_comparison(self):
        values=self.records()
        mutations=[values[:-1],values+[values[0]],deepcopy(values),deepcopy(values),deepcopy(values)]
        mutations[2][1]['run_id']=mutations[2][0]['run_id']
        mutations[3][0]['round']=2
        mutations[4][0]['phase']='diagnostic'
        for rows_ in mutations:
            with self.assertRaises(ValueError):c1.summarize(rows_)

    def test_comparison_identity_mismatch_rejected(self):
        values=self.records();values[-1]['comparison_key']='different'
        with self.assertRaises(ValueError):c1.summarize(values)


class C1ExecutionTests(unittest.TestCase):
    def run_fixture(self,fail_at=None,interrupt=False,budget=None):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            d=Path(temp);conf=config(output=str(d/'suite'))
            for name in ('hardware','runtime'):(d/(name+'.json')).write_text('{}')
            plan={'reference_sha256':'fixture','hardware_baseline':str(d/'hardware.json'),
                  'runtime_baseline':str(d/'runtime.json')}
            seen=[]
            def process(argv,directory,environment,preflight,samplers,validate,limits,**kwargs):
                seen.append({'argv':argv,'baseline':preflight.args[-1],'output':directory,'limits':limits})
                Path(directory).mkdir()
                policy=c1.BY_ID[list('EDABDBEABADE')[len(seen)-1]]
                metrics=c1.parse_benchmark(timed_rows(conf,policy),conf,3,load_mode='none')
                if interrupt:os.kill(os.getpid(),signal.SIGTERM)
                return {'status':'failed' if len(seen)==fail_at else 'passed',
                        'reason':'fixture_failure' if len(seen)==fail_at else None,'launches':1,
                        'validation':{'metrics':metrics},'cleanup_complete':True,'reader_cleanup_complete':True}
            stack.enter_context(patch.object(c1,'supervise',side_effect=process))
            stack.enter_context(patch.object(c1,'cooldown',return_value={'status':'ready','reader_cleanup_complete':True}))
            if budget is not None:stack.enter_context(patch.object(c1,'SUITE_SECONDS',budget))
            result=c1.execute(conf,plan)
            self.assertEqual(json.loads((Path(conf.output)/'suite-result.json').read_text()),result)
            self.assertIn('\"event\": \"final\"',(Path(conf.output)/'suite-events.jsonl').read_text())
            return result,seen

    def test_twelve_independent_launches_exact_order_and_frozen_baseline(self):
        result,seen=self.run_fixture()
        self.assertEqual(result['status'],'passed',result)
        self.assertEqual(result['launches'],12)
        self.assertEqual([a['id'] for a in result['attempts']],list('EDABDBEABADE'))
        self.assertEqual(len({s['output'] for s in seen}),12)
        self.assertEqual(len({s['baseline'] for s in seen}),1)
        self.assertEqual(result['comparison']['independent_n_per_configuration'],3)

    def test_every_failure_position_stops_without_retry_or_partial_comparison(self):
        for n in range(1,13):
            with self.subTest(n=n):
                result,seen=self.run_fixture(fail_at=n)
                self.assertEqual(result['status'],'failed')
                self.assertEqual(len(seen),n);self.assertIsNone(result['comparison'])
                self.assertEqual(len(result['measurements']),n-1)

    def test_signal_deadline_and_existing_output_refuse_further_launches(self):
        result,seen=self.run_fixture(interrupt=True)
        self.assertEqual(result['status'],'failed');self.assertEqual(len(seen),1)
        result,seen=self.run_fixture(budget=899)
        self.assertEqual(result['status'],'failed');self.assertEqual(len(seen),0)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):c1.execute(config(output=temp),{})

    def test_real_cpu_child_quiet_validation_monitoring_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ready=root/'ready';directory=root/'attempt'
            conf=config(output=str(directory));policy=c1.BY_ID['E']
            profile={'gpu':{d:{'median_delta_bytes':20*GIB,'peak_delta_bytes':20*GIB}
                            for d in ('CUDA0','ROCm0')},'c0_peak_sampled_hwm_bytes':32*1024**2,
                     'host_estimated_bytes':1024**2,'layers':{str(i):'CUDA0' if i<41 else 'ROCm0' for i in range(81)}}
            def samplers(pid):
                def fast():
                    r=fast_sample(pid);r['memory']['MemAvailable']=64*GIB
                    r['memory']['HardwareCorrupted']=0;r['swap_used']=0
                    return r
                def gpu():
                    values=gpu_fixture()
                    for v in values.values():
                        v['utilization_percent']=50 if ready.exists() else 0
                        if ready.exists():v['used_bytes']+=20*GIB
                    return values
                return {'thermal':thermal_fixture,'fast':fast,'gpu':gpu,'maps':lambda:maps_sample(pid,'/synthetic')}
            code=f'import time;from pathlib import Path;Path({str(ready)!r}).write_text("ready");print({json.dumps(timed_rows(conf,policy))!r},flush=True);time.sleep(4.2)'
            result=supervise([sys.executable,'-c',code],directory,c1.environment(conf),
                             lambda:{'swap_baseline':0,'c0_reference':{'profile':profile}},samplers,
                             lambda out,err:c1.validate(conf,policy,out,err),
                             replace(c1.LIMITS,overall=8,preflight=2,term_grace=.05))
            self.assertEqual(result['status'],'passed',result)
            self.assertTrue(result['cleanup_complete']);self.assertTrue(result['reader_cleanup_complete'])
            self.assertEqual(result['validation']['metrics']['pp']['n'],3)
            self.assertIsNone(result['validation']['placement_verification']['fresh_observed_layer_map'])
