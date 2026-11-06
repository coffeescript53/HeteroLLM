"""CPU-only U/T contracts, repeated phases, paired statistics and logit rules."""
from array import array
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
from dataclasses import asdict,replace
import io,json,math,os,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from helpers import ROOT
from heterollm import stage_c2_calibration as c,calibration_analysis as a
from heterollm.timing_analysis import analyze_timing
from test_stage_c2_timing import evidence
from test_stage_c import rows as benchmark_rows
import test_supervisor as support


def observations():
    return [{'block':b,'arm':arm,'run_id':f'{b}-{arm}','status':'passed','contract_sha256':'a'*64,
             'samples_ns':{'pp512':[100,110,120] if arm=='U' else [110,121,132],
                           'tg128':[200,220,240] if arm=='U' else [220,242,264]}}
            for b,order in enumerate(a.ORDERS,1) for arm in order]


def repeated_evidence():
    t,cpu=evidence();tr=[json.loads(x) for x in t.splitlines()];cr=[json.loads(x) for x in cpu.splitlines()]
    output=[tr[0]];spans=[];count=0
    for p,(name,rep) in enumerate([('warmup_pp',0),('pp512',0),('pp512',1),('pp512',2),('warmup_tg',0),('tg128',0),('tg128',1),('tg128',2)],1):
        delta=(p-1)*100000
        for row in tr[1:]:
            if row.get('phase')!=1:continue
            r=dict(row,phase=p)
            for key in ('cpu_ns','start_before','start_after','end_before','end_after','observed'):
                if key in r:r[key]+=delta
            if r['type']=='phase_begin':r.update(name=name,repetition=rep)
            if r['type']=='device':r.update(id=count,graph=p);count+=1
            output.append(r)
        for row in cr[1:-1]:
            if row.get('graph')!=1:continue
            spans.append(dict(row,seq=len(spans),graph=p,begin_ns=row['begin_ns']+delta,end_ns=row['end_ns']+delta))
    output.append(dict(tr[-1],phases=8,started=count,completed=count))
    return '\n'.join(map(json.dumps,output)), '\n'.join(map(json.dumps,[cr[0],*spans,dict(cr[-1],recorded=len(spans))]))

class StatisticsTests(unittest.TestCase):
    def test_paired_process_means_uncertainty_and_counterbalance(self):
        result=a.paired_summary(observations())
        for phase in ('pp512','tg128'):
            s=result['workloads'][phase];self.assertAlmostEqual(s['geometric_overhead_percent'],10)
            self.assertEqual(s['independent_pairs'],6);self.assertEqual(len(s['pairs']),6)
            self.assertEqual(s['by_order']['UT']['n'],3);self.assertAlmostEqual(s['sd_log_elapsed_ratio'],0)
            self.assertAlmostEqual(s['nominal_95pct_ci_percent'][0],10)
    def test_partial_duplicate_wrong_order_contract_repetitions_and_nonfinite_rejected(self):
        variants=[];v=observations();variants.append(v[:-1]);variants.append(v+[v[0]])
        for key,value in [('run_id','1-U'),('contract_sha256','b'*64),('status','failed'),('block',99)]:
            v=observations();v[1][key]=value;variants.append(v)
        for samples in ([1,2],[1,2,0],[1,True,3],[1,float('nan'),3]):
            v=observations();v[0]['samples_ns']['pp512']=samples;variants.append(v)
        variants.append(list(reversed(observations())))
        for v in variants:
            with self.assertRaises(ValueError):a.paired_summary(v)
    def test_zero_and_negative_effects_are_valid(self):
        for factor in (1,.9):
            values=observations()
            for r in values:
                if r['arm']=='T':r['samples_ns']={k:[round(v*factor) for v in vs] for k,vs in {'pp512':[100,110,120],'tg128':[200,220,240]}.items()}
            self.assertAlmostEqual(a.paired_summary(values)['workloads']['pp512']['geometric_overhead_percent'],(factor-1)*100)

class RepetitionTests(unittest.TestCase):
    def test_eight_phases_complete_with_three_repetitions(self):
        t,cpu=repeated_evidence();r=analyze_timing(t,cpu,'',repetitions=3)
        self.assertEqual(len(r['phases']),8);self.assertEqual(r['device_records'],16)
        self.assertEqual(r['phases'][4]['repetition'],2)
        with self.assertRaises(ValueError):analyze_timing(t,cpu,'')
    def test_wrong_repetition_missing_phase_footer_and_overflow_rejected(self):
        t,cpu=repeated_evidence();rows=[json.loads(s) for s in t.splitlines()]
        for mutate in ('repeat','phase','footer','truncated','overflow'):
            v=deepcopy(rows)
            if mutate=='repeat':next(r for r in v if r.get('phase')==3 and r['type']=='phase_begin')['repetition']=0
            if mutate=='phase':next(r for r in v if r['type']=='phase_begin')['name']='pp512'
            if mutate=='footer':v[-1]['phases']=4
            if mutate=='truncated':v.pop()
            if mutate=='overflow':v[-1]['invalid']=True
            with self.assertRaises(ValueError):analyze_timing('\n'.join(map(json.dumps,v)),cpu,'',repetitions=3)
        for n in (True,0,2,4):
            with self.assertRaises(ValueError):analyze_timing(t,cpu,'',repetitions=n)

class NumericalTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.patch=patch.object(a,'VOCAB',4);self.patch.start();self.addCleanup(self.patch.stop)
        for arm in ('U','T'):
            d=self.root/arm;d.mkdir()
            rows=[{'type':'header','schema':'heterollm.equivalence.v1','arm':arm,'vocabulary':4,'rows':129,'dtype':'little-endian-float32'}]
            rows += [dict(type='row',row=i,phase='pp512' if i==0 else 'tg128',position=511 if i==0 else i-1,argmax=0,piece_hex='61') for i in range(129)]
            rows += [{'type':'footer','rows':129,'complete':True}]
            (d/'predictions.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            (d/'logits.f32').write_bytes(array('f',[3.,2.,1.,0.]*129).tobytes())
    def compare(self):return a.compare_captures(self.root/'U',self.root/'T')
    def test_exact_and_tolerated_full_logits(self):
        self.assertTrue(self.compare()['passed'])
        path=self.root/'T/logits.f32';v=array('f');v.frombytes(path.read_bytes());v[2]+=1e-5;path.write_bytes(v.tobytes())
        self.assertTrue(self.compare()['passed']);self.assertEqual(self.compare()['different_values'],1)
    def test_large_deviation_prediction_or_piece_mismatch_fails(self):
        path=self.root/'T/logits.f32';v=array('f');v.frombytes(path.read_bytes());v[2]+=.01;path.write_bytes(v.tobytes())
        self.assertFalse(self.compare()['passed'])
        path.write_bytes(array('f',[3.,2.,1.,0.]*129).tobytes())
        p=self.root/'T/predictions.jsonl';p.write_text(p.read_text().replace('"piece_hex": "61"','"piece_hex": "62"'))
        self.assertFalse(self.compare()['passed'])
    def test_missing_truncated_nan_argmax_footer_and_symlink_fail(self):
        p=self.root/'T/logits.f32';original=p.read_bytes()
        for raw in (original[:-1],original+original,array('f',[float('nan')]+[0.]*(4*129-1)).tobytes()):
            p.write_bytes(raw)
            with self.assertRaises(ValueError):self.compare()
        p.write_bytes(original);meta=self.root/'T/predictions.jsonl';text=meta.read_text()
        for raw in (text.rsplit('\n',1)[0],text.replace('"argmax": 0','"argmax": 1'),text.replace('"complete": true','"complete": false'),text.replace('"complete": true','"complete": 1'),text.replace('"rows": 129','"rows": 129.0'),text.replace('"row": 0','"row": false')):
            meta.write_text(raw)
            with self.assertRaises(ValueError):self.compare()
        meta.write_text(text);p.unlink();p.symlink_to(self.root/'U/logits.f32')
        with self.assertRaises(ValueError):self.compare()
    def test_full_capture_recomputes_argmax_and_hashes(self):
        r=a.validate_capture(self.root/'U','U');self.assertEqual(r['rows'],129);self.assertTrue(r['complete'])
        self.assertEqual(len(r['files']['logits.f32']['sha256']),64)

class ContractTests(unittest.TestCase):
    def test_new_timed_validation_accepts_eight_phases_and_rejects_incomplete_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);conf=c.configuration('T',out)
            t,cpu=repeated_evidence();(out/'device-timing.jsonl').write_text(t);(out/'cpu-trace.jsonl').write_text(cpu)
            (out/'preflight.json').write_text(json.dumps({'c0_reference':{'profile':{}}}))
            paths=[str((Path(conf.bench).parent/name).resolve()) for name in c.timing.REQUIRED_LIBS]
            (out/'maps.jsonl').write_text(json.dumps({'data':{'loaded_runtime_libraries':paths}})+'\n')
            rows=benchmark_rows(conf,c.POLICY)
            for row in rows:row['samples_ns']=[1000000000]*3
            with patch.object(c.c1,'allocation_check',return_value={}),patch.object(c,'runtime_files',return_value=[{'path':p} for p in paths]):
                value=c.validate(conf,'T','overhead',json.dumps(rows),'');self.assertEqual(value['errors'],[])
                self.assertTrue(value['timing_complete']);self.assertEqual(len(value['samples_ns']['pp512']),3)
                (out/'device-timing.jsonl').write_text(t.rsplit('\n',1)[0])
                self.assertTrue(c.validate(conf,'T','overhead',json.dumps(rows),'')['errors'])
                (out/'device-timing.jsonl').write_text(t);rows[0]['n_prompt']=256
                self.assertTrue(c.validate(conf,'T','overhead',json.dumps(rows),'')['errors'])
    def test_u_rejects_trace_artifact_and_foreign_runtime_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);conf=c.configuration('U',out);rows=benchmark_rows(conf,c.POLICY)
            for row in rows:row['samples_ns']=[1000000000]*3
            (out/'preflight.json').write_text(json.dumps({'c0_reference':{'profile':{}}}))
            paths=[str((Path(conf.bench).parent/name).resolve()) for name in c.timing.REQUIRED_LIBS]
            (out/'maps.jsonl').write_text(json.dumps({'data':{'loaded_runtime_libraries':paths}})+'\n')
            with patch.object(c.c1,'allocation_check',return_value={}),patch.object(c,'runtime_files',return_value=[{'path':p} for p in paths]):
                self.assertEqual(c.validate(conf,'U','overhead',json.dumps(rows),'')['errors'],[])
                (out/'cpu-trace.jsonl').write_text('unexpected')
                self.assertTrue(c.validate(conf,'U','overhead',json.dumps(rows),'')['errors'])
                (out/'cpu-trace.jsonl').unlink()
                (out/'maps.jsonl').write_text(json.dumps({'data':{'loaded_runtime_libraries':['/unapproved/libllama.so']}})+'\n')
                self.assertTrue(c.validate(conf,'U','overhead',json.dumps(rows),'')['errors'])
    def test_process_wall_time_is_not_mislabeled_as_loading(self):
        with patch.object(c.c1,'read_records',return_value=[{'event':'exec_release','monotonic':10},{'event':'reaped','monotonic':14}]):
            value=c.process_timing('unused');self.assertEqual(value['exec_release_to_reaped_seconds'],4)
            self.assertIsNone(value['isolated_model_loading_seconds'])
        with patch.object(c.c1,'read_records',return_value=[]):
            with self.assertRaises(ValueError):c.process_timing('unused')
    def test_harness_cross_runtime_or_missing_library_is_rejected(self):
        conf=c.configuration('T','unused')
        for output in ('libllama.so => /example/HeteroLLM/llama.cpp/build-mixed/bin/libllama.so (0x0)',
                       'libllama.so => not found','libc.so => /lib/libc.so (0x0)'):
            with patch.object(c,'query',return_value={'exit_code':0,'stdout':output,'stderr':''}):
                with self.assertRaises(ValueError):c.build_dependencies('unused',conf)
    def test_wrong_harness_identity_stops_before_model_preflight(self):
        plan=c.expected_plan('equivalence');exact=c.manifest(plan)
        with tempfile.TemporaryDirectory() as temp:
            launch=deepcopy(exact['launches'][0]);launch['configuration']['output']=temp
            with patch.object(c,'manifest',return_value=exact),patch.object(c,'build_dependencies',return_value={'executable':'wrong','dependencies':{}}),patch.object(c.c1,'preflight') as pre:
                with self.assertRaisesRegex(ValueError,'harness/dependency'):c.preflight(plan,exact,launch,0)
                pre.assert_not_called()
    def test_exact_commands_arms_controls_order_and_safety(self):
        plan=c.expected_plan('overhead');m=c.manifest(plan)
        self.assertEqual(''.join(x['arm'] for x in m['launches']),'UTTUUTTUUTTU')
        self.assertEqual(m['limits'],asdict(c.c0.C0_LIMITS));self.assertEqual(m['suite_seconds'],11400)
        self.assertEqual(m['maximum_model_processes'],12)
        for row in m['launches']:
            argv=row['argv'];env=row['environment']
            for flag,value in {'-r':'3','-p':'512','-n':'128','-d':'0','-lm':'none','-lzm':'off','-ts':'1/1','-ngl':'999','-fitt':'0'}.items():self.assertEqual(argv[argv.index(flag)+1],value)
            self.assertNotIn('-v',argv);self.assertNotIn('GGML_SCHED_DEBUG',env);self.assertNotIn('LD_PRELOAD',env)
            self.assertEqual('HETEROLLM_TIMING_FILE' in env,row['arm']=='T')
            self.assertIn('runtime-c2-phase-v3' if row['arm']=='T' else 'llama.cpp',argv[0])
    def test_numerical_separate_two_processes_and_unmodified_runtime(self):
        m=c.manifest(c.expected_plan('equivalence'))
        self.assertEqual(m['maximum_model_processes'],2);self.assertEqual(m['suite_seconds'],3000)
        self.assertEqual([x['arm'] for x in m['launches']],['U','T'])
        self.assertEqual(m['numerical_contract']['atol'],1e-4)
        self.assertNotEqual(m['launches'][0]['argv'][0],m['launches'][1]['argv'][0])
    def test_old_plan_mutation_and_source_mismatch_rejected(self):
        for path in c.BASE_PLANS.values():
            with self.assertRaises(ValueError):c.load_plan(path)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'plan.json';plan=c.expected_plan('overhead');p.write_text(json.dumps(plan))
            with patch.object(c,'manifest',return_value={'changed':1}):
                with self.assertRaises((ValueError,FileNotFoundError)):c.load_plan(p)
            plan['output']+='/changed';p.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):c.load_plan(p)
    def run_suite(self,bad=None,kind='overhead',deadline=False):
        with tempfile.TemporaryDirectory() as temp:
            plan=c.expected_plan(kind);plan['output']=str(Path(temp)/'suite')
            exact=c.manifest(plan);called=[]
            def fake(argv,output,env,*args,**kwargs):
                Path(output).mkdir();called.append(output)
                return {'status':'failed' if bad=='run' else 'passed','launches':1,'cleanup_complete':bad!='cleanup',
                    'reader_cleanup_complete':bad!='reader','elapsed_seconds':901 if bad=='timeout' else 1,
                    'validation':{'samples_ns':{'pp512':[100,100,100],'tg128':[100,100,100]}}}
            with patch.object(c.c1,'cooldown',return_value={'status':'failed' if bad=='cooldown' else 'ready','reader_cleanup_complete':bad!='coolreader'}),patch.object(c,'supervise',side_effect=fake),patch.object(c.c0,'resource_summary',return_value={}),patch.object(c,'process_timing',return_value={}),patch.dict(c.TOTALS,{kind:0 if deadline else c.TOTALS[kind]}),patch.object(a,'compare_captures',return_value={'passed':bad!='comparison'}):
                result=c.execute(plan,exact)
            saved=json.loads((Path(plan['output'])/'suite-result.json').read_text())
            self.assertEqual(result,saved)
            return result,len(called)
    def test_twelve_independent_or_stop_without_retry_and_durable_traceback(self):
        r,n=self.run_suite();self.assertEqual((r['status'],n),('passed',12))
        for reason in ('run','cleanup','reader','timeout'):
            r,n=self.run_suite(reason);self.assertEqual((r['status'],n),('failed',1));self.assertIn('traceback',r)
        for reason in ('cooldown','coolreader'):
            r,n=self.run_suite(reason);self.assertEqual((r['status'],n),('failed',0))
        r,n=self.run_suite(deadline=True);self.assertEqual((r['status'],n),('failed',0))
    def test_numerical_failure_and_scope_have_no_followup(self):
        r,n=self.run_suite(kind='equivalence');self.assertEqual((r['status'],n),('passed',2))
        r,n=self.run_suite('comparison',kind='equivalence');self.assertEqual((r['status'],n),('failed',2))
    def test_each_preflight_dispatches_exact_arm_checker(self):
        plan=c.expected_plan('overhead');m=c.manifest(plan)
        with patch.object(c,'manifest',return_value=m),patch.object(c.c1,'preflight',return_value='U') as u,patch.object(c.timing,'preflight',return_value='T') as t:
            for row in m['launches'][:2]:self.assertEqual(c.preflight(plan,m,row,7),row['arm'])
            self.assertEqual(u.call_count,1);self.assertEqual(t.call_count,1)
        with patch.object(c,'manifest',return_value={'changed':1}):
            with self.assertRaises(ValueError):c.preflight(plan,m,m['launches'][0],0)
    def test_real_supervisor_cleanup_on_timeout_and_invalid_capture(self):
        result,_=support.SupervisorTests().run_case(code='time.sleep(10)',limits=replace(c.LIMITS,overall=.8,preflight=.4,term_grace=.05))
        self.assertEqual(result['status'],'failed');self.assertTrue(result['cleanup_complete']);self.assertTrue(result['reader_cleanup_complete'])
        result,_=support.SupervisorTests().run_case(validate_fn=lambda o,e:{'errors':['synthetic invalid capture']})
        self.assertEqual(result['exit_code'],0);self.assertEqual(result['status'],'failed');self.assertTrue(result['cleanup_complete'])
    def test_preview_does_not_run_or_discover(self):
        plan=c.expected_plan('overhead')
        with patch.object(c,'load_plan',return_value=(plan,{})),patch.object(c,'execute',side_effect=AssertionError('GPU disallowed')),redirect_stdout(io.StringIO()) as output:
            self.assertEqual(c.main(['--plan','fixture']),0)
            self.assertEqual(json.loads(output.getvalue())['mode'],'preview_only')

if __name__=='__main__':unittest.main()
