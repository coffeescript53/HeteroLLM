"""Reuse all reviewed adapter contract tests against the isolated v3 contract."""
import importlib.util,json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from helpers import ROOT
from heterollm.stage_c2_timing import adapter as timing
from heterollm.timing_analysis import HEADER,PHASES
from test_c2_trace_v2 import fixture_v2
from test_stage_c import rows
import test_supervisor as support

spec=importlib.util.spec_from_file_location('timing_contract_tests',ROOT/'tests/test_stage_c2.py')
contracts=importlib.util.module_from_spec(spec);spec.loader.exec_module(contracts);contracts.c2=timing
class TimingContractTests(contracts.C2ContractTests):pass

def evidence():
    base=fixture_v2();spans=[];records=[dict(HEADER)];ids=0
    for phase,name in enumerate(PHASES,1):
        start=phase*100000
        records.append(dict(type='phase_begin',phase=phase,name=name,repetition=0,cpu_ns=start))
        for split,backend in enumerate(('CUDA0','ROCm0')):
            t=start+split*10000+100
            spans.append(dict(base[1],seq=len(spans),thread=7,graph=phase,split=split,kind='graph.submit',src=backend,dst='',logical_bytes=0,begin_ns=t,end_ns=t+10))
            records.append(dict(type='device',id=ids,phase=phase,graph=phase,split=split,thread=7,logical_bytes=0,start_before=t+1,start_after=t+2,end_before=t+3,end_after=t+4,observed=t+20,elapsed_ms=.000001,kind='graph_uncaptured',backend=backend,stream='main'));ids+=1
        spans.append(dict(base[1],seq=len(spans),thread=7,graph=phase,split=-1,kind='graph.compute_splits',src='',dst='',logical_bytes=0,begin_ns=start+50,end_ns=start+20000))
        records.append(dict(type='phase_end',phase=phase,cpu_ns=start+30000))
    records.append(dict(type='footer',phases=4,started=ids,completed=ids,pending=0,active_phase=0,invalid=False,complete=True))
    cpu=[base[0],*spans,dict(base[-1],recorded=len(spans))]
    return '\n'.join(map(json.dumps,records))+'\n','\n'.join(map(json.dumps,cpu))+'\n'

class TimingAcceptanceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.config=replace(timing.expected_config(),output=str(self.path))
        (self.path/'preflight.json').write_text(json.dumps({'c0_reference':{'profile':{}}}))
        (self.path/'maps.jsonl').write_text(json.dumps({'data':{'loaded_runtime_libraries':[str((timing.BIN/n).resolve()) for n in timing.REQUIRED_LIBS]}})+'\n')
        t,c=evidence();(self.path/'device-timing.jsonl').write_text(t);(self.path/'cpu-trace.jsonl').write_text(c)
        self.stdout=json.dumps(rows(self.config,timing.POLICY))
        p=patch.object(timing.c1,'allocation_check',return_value={'status':'allocation_consistent_with_C0'});p.start();self.addCleanup(p.stop)
    def test_complete_zero_fallback_valid_and_extra_environment_exact(self):
        value=timing.validate(self.config,self.stdout,'');self.assertEqual(value['errors'],[])
        self.assertEqual(value['completed_device_timing']['matched_host_stage_legs'],0)
        self.assertIsNone(value['gpu_execution_time']);self.assertIsNone(value['gpu_overlap'])
        self.assertEqual(timing.environment(self.config)['HETEROLLM_TIMING_FILE'],str(self.path/'device-timing.jsonl'))
    def test_bad_device_trace_fails_after_exit_zero_and_cleans_up(self):
        (self.path/'device-timing.jsonl').write_text('{bad}')
        result,saved=support.SupervisorTests().run_case(code='print('+repr(self.stdout)+')',validate_fn=lambda out,err:timing.validate(self.config,out,err))
        self.assertEqual(result['exit_code'],0);self.assertEqual(result['status'],'failed');self.assertTrue(result['cleanup_complete']);self.assertTrue(result['reader_cleanup_complete'])
    def test_wrong_workload_and_v2_runtime_maps_rejected(self):
        bad=rows(self.config,timing.POLICY);bad[0]['n_prompt']=256
        self.assertTrue(timing.validate(self.config,json.dumps(bad),'')['errors'])
        (self.path/'maps.jsonl').write_text(json.dumps({'data':{'loaded_runtime_libraries':['/example/HeteroLLM/runtime-c2-trace-v2/build-mixed/bin/libggml-base.so']}})+'\n')
        self.assertTrue(timing.validate(self.config,self.stdout,'')['errors'])
    def test_missing_symlink_and_truncated_timing_fail(self):
        p=self.path/'device-timing.jsonl';p.unlink()
        self.assertTrue(timing.validate(self.config,self.stdout,'')['errors'])
        p.symlink_to(self.path/'cpu-trace.jsonl');self.assertTrue(timing.validate(self.config,self.stdout,'')['errors'])

if __name__=='__main__':unittest.main()
