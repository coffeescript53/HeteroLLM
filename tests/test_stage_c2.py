"""C2 contract/acceptance tests. Synthetic data and Python children only."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helpers import config, ROOT
from heterollm import stage_c as c0, stage_c1 as c1, stage_c2 as c2
from test_c2_trace_v2 import fixture_v2 as fixture
from test_stage_c import rows, hardware
import test_supervisor as support


def valid_trace():
    values=fixture();template=values[1]
    events=[dict(template,seq=i,kind=kind,src=src,dst='',logical_bytes=0)
            for i,(kind,src) in enumerate([('graph.compute_splits',''),('graph.submit','CUDA0'),('graph.submit','ROCm0')])]
    return [values[0],*events,dict(values[-1],recorded=len(events))]


class C2AcceptanceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.config=replace(c2.expected_config(),output=str(self.path))
        (self.path/'preflight.json').write_text(json.dumps({'c0_reference':{'profile':{}}}))
        (self.path/'maps.jsonl').write_text(json.dumps({'data':{'loaded_runtime_libraries':
            [str((c2.BIN/n).resolve()) for n in c2.REQUIRED_LIBS]}})+'\n')
        self.trace=valid_trace();self.save_trace()
        self.stdout=json.dumps(rows(self.config,c2.POLICY))
        p=patch.object(c1,'allocation_check',return_value={'status':'allocation_consistent_with_C0'})
        p.start();self.addCleanup(p.stop)

    def save_trace(self):
        (self.path/'cpu-trace.jsonl').write_text('\n'.join(map(json.dumps,self.trace))+'\n')

    def validate(self,stderr='',stdout=None):
        return c2.validate(self.config,self.stdout if stdout is None else stdout,stderr)

    def test_valid_zero_host_staging_is_an_accepted_negative_finding(self):
        value=self.validate();self.assertEqual(value['errors'],[])
        self.assertEqual(value['trace']['host_staged_directions'],{})
        self.assertIsNone(value['gpu_execution_time']);self.assertIsNone(value['phase_attribution'])
        self.assertFalse(value['timings_eligible_for_comparison'])

    def test_missing_malformed_overflow_truncation_errors_and_missing_backend_rejected(self):
        original=valid_trace()
        for change in ('missing','json','header_only','footer_missing','overflow','unfinished',
                       'metadata_truncated','error','one_backend','wrong_capacity','bad_span','duplicate','old_schema','aggregate_loss'):
            with self.subTest(change=change):
                self.trace=deepcopy(original);self.save_trace();stderr=''
                target=self.path/'cpu-trace.jsonl'
                if change=='missing':target.unlink()
                elif change=='json':target.write_text('{bad json\n')
                elif change=='header_only':target.write_text(json.dumps(self.trace[0]))
                elif change=='footer_missing':self.trace.pop();self.save_trace()
                elif change in ('overflow','unfinished','metadata_truncated'):
                    self.trace[-1]['dropped' if change=='overflow' else change]=1;self.save_trace()
                elif change=='old_schema':self.trace[0]['schema']='heterollm.cpu_trace.v1';self.save_trace()
                elif change=='aggregate_loss':self.trace[-1]['aggregate_dropped']=1;self.save_trace()
                elif change=='error':stderr='HETEROLLM_TRACE_ERROR: flush failed'
                elif change=='one_backend':self.trace[3]['src']='CUDA0';self.save_trace()
                elif change=='wrong_capacity':self.trace[0]['capacity']=16;self.save_trace()
                elif change=='bad_span':self.trace[1]['end_ns']=-1;self.save_trace()
                elif change=='duplicate':target.write_text(target.read_text().replace('"type": "header"','"type": "header", "type": "header"'))
                self.assertTrue(any(e.startswith('trace:') for e in self.validate(stderr)['errors']))

    def test_wrong_workloads_and_runtime_maps_rejected(self):
        for field,value in [('n_depth',1),('n_prompt',256),('load_mode','mmap'),('samples_ns',[1,2]),('tensor_split','2/1')]:
            changed=rows(self.config,c2.POLICY);changed[0][field]=value
            self.assertTrue(self.validate(stdout=json.dumps(changed))['errors'],field)
        (self.path/'maps.jsonl').write_text(json.dumps({'data':{'loaded_runtime_libraries':[]}})+'\n')
        self.assertTrue(self.validate()['errors'])

    def test_exit_zero_with_bad_trace_fails_and_cleans_up(self):
        (self.path/'cpu-trace.jsonl').unlink()
        result,saved=support.SupervisorTests().run_case(
            code='print('+repr(self.stdout)+')',
            validate_fn=lambda out,err:c2.validate(self.config,out,err))
        self.assertEqual(result['exit_code'],0);self.assertEqual(result['status'],'failed')
        self.assertTrue(result['validation']['errors']);self.assertTrue(result['cleanup_complete'])
        self.assertTrue(result['reader_cleanup_complete']);self.assertIn('wait4-rusage.json',saved)


class C2ContractTests(unittest.TestCase):
    def test_exact_one_process_controls_and_original_safety_limits(self):
        conf=c2.expected_config();argv=c2.command(conf)
        expected=c0.command(conf,c2.POLICY);expected.remove('-v');self.assertEqual(argv,expected)
        for flag,value in {'-r':'1','-p':'512','-n':'128','-d':'0','-lm':'none','-lzm':'off','-ts':'1/1'}.items():
            self.assertEqual(argv[argv.index(flag)+1],value)
        self.assertEqual(asdict(c2.LIMITS),asdict(c0.C0_LIMITS));self.assertEqual(c2.SUITE_SECONDS,1500)
        self.assertEqual(conf.max_launches,1)
        self.assertNotIn('GGML_SCHED_DEBUG',c2.environment(conf));self.assertNotIn('LD_PRELOAD',c2.environment(conf))

    def test_candidate_wrong_source_runtime_dependencies_build_fail_closed(self):
        approved={'source':{'status':'dirty'},'runtime':['candidate'],'dependencies':{'files':{'x':1}},'build_files':{'x':2}}
        with tempfile.TemporaryDirectory() as temp:
            receipt=Path(temp)/'receipt.json';receipt.write_text(json.dumps(approved))
            for key in ('source','runtime','dependencies','build_files'):
                actual=deepcopy(approved)
                actual[key]={'files':{'wrong':1}} if key=='dependencies' else 'wrong'
                with patch.object(c2,'build_receipt',return_value=actual),patch.object(c2,'discover') as discover:
                    with self.assertRaisesRegex(ValueError,'mismatch'):
                        c2.verify_candidate(config(),{'runtime_receipt':str(receipt)},lambda n,v:v)
                    discover.assert_not_called()

    def test_original_runtime_check_still_rejects_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline=Path(temp)/'baseline';baseline.write_text('["original"]')
            with patch.object(c0,'runtime_identity',return_value=['candidate']):
                with self.assertRaisesRegex(ValueError,'runtime files changed'):
                    c0.runtime_preflight(config(),{'runtime_baseline':str(baseline)},lambda n,v:v)

    def test_dependency_resolution_refuses_original_external_or_unresolved_ggml(self):
        conf=c2.expected_config()
        for output in ('libggml-base.so => '+str(ROOT/'llama.cpp/build-mixed/bin/libggml-base.so')+' (0x0)',
                       'libggml-base.so => /tmp/other/libggml-base.so (0x0)',
                       'libggml-base.so => not found'):
            with patch.object(c2,'query',return_value={'exit_code':0,'stdout':output,'stderr':''}):
                with self.assertRaises(ValueError):c2.candidate_dependencies(conf)

    def test_manifest_plan_tampering_and_old_stage_plans_are_refused(self):
        for name in ('stage-c0-plan.json','stage-c1-memtotal-fixed-plan.json','stage-c2-routing-plan.json'):
            with self.assertRaises(ValueError):c2.load_plan(ROOT/'configs'/name)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);conf=c2.expected_config()
            configuration=path/'config.json';configuration.write_text(json.dumps(asdict(conf)))
            plan={'stage':'C2_ROUTING','config':str(configuration),'receipt':'fixture','runtime_receipt':str(c2.RECEIPT),
                  'hardware_baseline':'fixture','reference':'fixture','reference_sha256':'x',
                  'mem_total_bytes':98774167552,'manifest':str(path/'manifest.json'),'authorization':'CPU fixture only'}
            (path/'manifest.json').write_text('{"fixture":1}')
            (path/'plan.json').write_text(json.dumps(plan))
            with patch.object(c2,'manifest',return_value={'fixture':1}),patch.object(c2,'sha256_file',return_value='x'):
                self.assertEqual(c2.load_plan(path/'plan.json')[0],conf)
                (path/'manifest.json').write_text('{"fixture":2}')
                with self.assertRaisesRegex(ValueError,'manifest'):c2.load_plan(path/'plan.json')
            configuration.write_text(json.dumps(asdict(replace(conf,max_launches=2))))
            with self.assertRaisesRegex(ValueError,'unreviewed C2 configuration'):c2.load_plan(path/'plan.json')

    def test_runtime_maps_reject_original_deleted_and_missing_read(self):
        conf=config()
        path=str(c2.BIN/'libggml-base.so');original=str(ROOT/'llama.cpp/build-mixed/bin/libggml-base.so')
        for value in (original,path+' (deleted)'):
            with patch.object(c2,'maps_sample',return_value={'model_mappings':[]}), \
                 patch.object(Path,'read_text',return_value='0000-1000 r-xp 0 0 0 '+value+'\n'):
                with self.assertRaisesRegex(ValueError,'unapproved loaded'):c2.runtime_maps(1,conf,{path})
        with patch.object(c2,'maps_sample',return_value={'model_mappings':[]}),patch.object(Path,'read_text',side_effect=PermissionError('maps')):
            with self.assertRaises(PermissionError):c2.runtime_maps(1,conf,{path})

    def test_uses_full_shared_preflight_with_separate_runtime_contract(self):
        mapping={str(i):'CUDA0' if i<41 else 'ROCm0' for i in range(81)}
        pre={'screening':{'estimated_layers':{'CUDA0':list(range(41)),'ROCm0':list(range(41,81))}}}
        with patch.object(c1,'verify_reference',return_value={'profiles':{'E':{'layers':mapping}}}), \
             patch.object(c0,'preflight',return_value=pre) as check:
            c2.preflight(config(),{},123)
            check.assert_called_once_with(config(),{},c2.POLICY,123,runtime_check=c2.verify_candidate)

    def run_execute(self,ready=None,run=None,deadline=False):
        with tempfile.TemporaryDirectory() as temp:
            conf=replace(c2.expected_config(),output=str(Path(temp)/'attempt'))
            exact={'child_output':str(Path(conf.output)/'child'),'argv':['unused'],'environment':{}}
            def child(*args,**kwargs):
                Path(exact['child_output']).mkdir()
                return run or {'status':'passed','launches':1,'cleanup_complete':True,'reader_cleanup_complete':True,'elapsed_seconds':1}
            with patch.object(c2,'manifest',return_value=exact),patch.object(c1,'cooldown',return_value=ready or {'status':'ready','reader_cleanup_complete':True}), \
                 patch.object(c2,'supervise',side_effect=child) as launch, \
                 patch.object(c0,'resource_summary',return_value={}),patch.object(c2,'SUITE_SECONDS',0 if deadline else 1500):
                result=c2.execute(conf,{})
                self.assertEqual(json.loads((Path(conf.output)/'suite-result.json').read_text()),result)
                return result,launch.call_count

    def test_exactly_one_and_no_retry_on_failure_or_bad_cleanup(self):
        value,count=self.run_execute();self.assertEqual((value['status'],count),('passed',1))
        for key in ('status','cleanup_complete','reader_cleanup_complete'):
            run={'status':'passed','launches':1,'cleanup_complete':True,'reader_cleanup_complete':True}
            run[key]='failed' if key=='status' else False
            value,count=self.run_execute(run=run);self.assertEqual((value['status'],count),('failed',1))
        for ready in ({'status':'failed','reader_cleanup_complete':True},{'status':'ready','reader_cleanup_complete':False}):
            value,count=self.run_execute(ready=ready);self.assertEqual((value['status'],count),('failed',0))

    def test_deadline_and_supervisor_timeout_keep_durable_failure_and_cleanup(self):
        value,count=self.run_execute(deadline=True);self.assertEqual((value['status'],count),('failed',0))
        self.assertIn('suite_deadline',value['error'])
        for elapsed in (901,-1):
            value,count=self.run_execute(run={'status':'passed','launches':1,'cleanup_complete':True,
                                              'reader_cleanup_complete':True,'elapsed_seconds':elapsed})
            self.assertEqual((value['status'],count),('failed',1))
            self.assertIn('attempt deadline',value['error'])
        result,saved=support.SupervisorTests().run_case(code='time.sleep(10)',
            limits=replace(c2.LIMITS,overall=.8,preflight=.4,term_grace=.05))
        self.assertEqual(result['status'],'failed');self.assertTrue(result['cleanup_complete'])
        self.assertTrue(result['reader_cleanup_complete']);self.assertIn('result.json',saved)

    def test_preview_never_discovers_hardware_or_runs_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            conf=replace(c2.expected_config(),output=str(Path(temp)/'new'))
            with patch.object(c2,'load_plan',return_value=(conf,{})),patch.object(c2,'manifest',return_value={}), \
                 patch.object(c2,'execute',side_effect=AssertionError('no execute')),patch.object(c2,'discover',side_effect=AssertionError('no hardware')),redirect_stdout(io.StringIO()) as out:
                c2.main(['--plan','fixture']);self.assertEqual(json.loads(out.getvalue())['mode'],'preview_only')
                Path(conf.output).mkdir()
                with self.assertRaisesRegex(ValueError,'already exists'):c2.main(['--plan','fixture','--execute'])
