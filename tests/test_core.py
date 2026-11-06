from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from heterollm import analysis
from heterollm.cli import main
from heterollm.config import Config,Policy,digest
from heterollm.gguf import ModelLayout,planning_placeholder,read_layout
from heterollm.hardware import clean_environment,parse_devices,parse_meminfo,parse_nvidia
from heterollm.planner import balanced_order,candidates,estimate
from heterollm.runtime import bench_command,functional_command,placement,response_correct,verify_placement
from helpers import ROOT,benchmark_rows,config,snapshot,tiny_gguf


class ConfigurationTests(unittest.TestCase):
    def test_relative_paths_follow_config_directory(self):
        self.assertEqual(config().source,str(ROOT/'llama.cpp'))
    def test_unknown_field_rejected(self):
        with self.assertRaises(TypeError): Config(**{**config().to_dict(),'surprise':True})
    def test_invalid_bounds(self):
        for kwargs in ({'max_launches':100},{'final_rounds':2},{'threads':True},{'cache_k':'q8_0'},{'model_sha256':'bad'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): config(**kwargs)
    def test_no_secret_or_scheduler_overrides(self):
        with self.assertRaises(ValueError): config(environment={'HF_TOKEN':'not-a-real-token'})
        with patch.dict('os.environ',{'GGML_SCHED_DEBUG':'2','HF_TOKEN':'test','LLAMA_ARG_MODEL':'x'}):
            self.assertNotIn('HF_TOKEN',clean_environment())
            self.assertNotIn('GGML_SCHED_DEBUG',clean_environment())
            self.assertNotIn('LLAMA_ARG_MODEL',clean_environment())
    def test_invalid_policy_ratios(self):
        for ratio in ((1,float('nan')),(0,1),(1,),(-1,2),(True,1)):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError): Policy('bad',('CUDA0','ROCm0'),ratio,80)
    def test_policy_roundtrip(self):
        p=Policy('mixed-equal',('CUDA0','ROCm0'),(1,1),999)
        self.assertEqual(p,Policy.from_dict(json.loads(json.dumps(p.to_dict()))))
    def test_unsupported_policy_rejected(self):
        with self.assertRaises(ValueError): Policy('x',('CUDA0',),None,10,split_mode='row')
        with self.assertRaises(ValueError): Policy('../escape',('CUDA0',),None,10)
        with self.assertRaises(ValueError): Policy('x',('CUDA0',),None,10,fit_target_mib=2048)


class HardwareTests(unittest.TestCase):
    def test_memory_units(self):
        m=parse_meminfo('MemTotal: 100 kB\nMemAvailable: 60 kB\nSwapTotal: 20 kB\nSwapFree: 20 kB\n')
        self.assertEqual(m['MemAvailable'],61440)
    def test_missing_memory_is_error(self):
        with self.assertRaises(ValueError): parse_meminfo('MemTotal: 0 kB')
    def test_devices_and_igpu(self):
        d=parse_devices('Available devices:\n  CUDA0: GPU (32088 MiB, 31000 MiB free)\n  ROCm0: R9700 (32624 MiB, 32500 MiB free)\n  ROCm1: APU (123 MiB, 100 MiB free)')
        self.assertEqual(len(d),3)
        self.assertEqual(d[1].free_bytes,32500*1024**2)
    def test_inconsistent_memory(self):
        with self.assertRaises(ValueError): parse_devices(' CUDA0: GPU (1 MiB, 2 MiB free)')
    def test_missing_sensor_is_not_zero(self):
        d=parse_nvidia('00000000:05:00.0, NVIDIA, 100, 200, 99, N/A, 595\n')
        self.assertIsNone(d[0]['temperature_c'])
        self.assertEqual(d[0]['pci'],'0000:05:00.0')


class ModelPlannerTests(unittest.TestCase):
    def test_header_only_reader(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tiny.gguf';tiny_gguf(p)
            layout=read_layout(p)
            self.assertEqual(layout.layer_bytes,(32,32,32))
            self.assertEqual(layout.host_bytes,32)
            self.assertEqual(layout.kv_bytes_per_layer_token,16)
    def test_truncated_model_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tiny.gguf';tiny_gguf(p);p.write_bytes(p.read_bytes()[:-2])
            with self.assertRaises(ValueError):read_layout(p)
    def test_bad_magic_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(b'nope'+b'\0'*30)
            with self.assertRaises(ValueError):read_layout(p)
    def test_six_deterministic_candidates(self):
        layout=replace(planning_placeholder(config().model_size_bytes),exact=True)
        a=candidates(layout,snapshot(),config())
        self.assertEqual([x.to_dict() for x in a],[x.to_dict() for x in candidates(layout,snapshot(),config())])
        self.assertEqual(len(a),6)
        self.assertTrue(all(set(x.policy.devices)<={'CUDA0','ROCm0'} for x in a))
        self.assertEqual(a[0].policy.name,'mixed-default')
    def test_single_gpu_offload_is_capacity_aware(self):
        layout=replace(planning_placeholder(config().model_size_bytes),exact=True)
        c=next(x for x in candidates(layout,snapshot(),config()) if x.policy.name=='cuda-only')
        self.assertTrue(c.admissible)
        self.assertLess(c.policy.gpu_layers,81)
        s=snapshot();s['devices'][0]['free_bytes']=20*1024**3
        smaller=next(x for x in candidates(layout,s,config()) if x.policy.name=='cuda-only')
        self.assertLess(smaller.policy.gpu_layers,c.policy.gpu_layers)
    def test_explicit_over_budget_is_rejected(self):
        layout=replace(planning_placeholder(80*1024**3),exact=True)
        c=estimate(Policy('x',('CUDA0','ROCm0'),(2,1),999),layout,snapshot(),config())
        self.assertFalse(c.admissible)
        self.assertTrue(any('exceed' in x for x in c.reasons))
    def test_layer_ratios_are_not_byte_ratios(self):
        layout=ModelLayout(100,(10,10,70),10,0)
        c=estimate(Policy('x',('CUDA0','ROCm0'),(1,1),999),layout,snapshot(),config())
        self.assertEqual(c.estimated_layers,{'CUDA0':[0,1],'ROCm0':[2]})
        self.assertEqual(c.gpu_estimated_bytes,{'CUDA0':20,'ROCm0':70})
    def test_host_memory_constraint(self):
        layout=ModelLayout(100*1024**3,(1,1),99*1024**3,0)
        self.assertFalse(estimate(Policy('x',('CUDA0',),None,2),layout,snapshot(),config()).admissible)
    def test_unverified_identity_rejected(self):
        s=snapshot();s['devices'][1]['pci']=None
        self.assertFalse(estimate(Policy('x',('CUDA0','ROCm0'),(1,1),999),ModelLayout(3,(1,1),1,0),s,config()).admissible)
    def test_placeholder_never_executes(self):
        self.assertTrue(all(not x.admissible for x in candidates(planning_placeholder(42_520_398_816),snapshot(),config())))
    def test_balanced_each_once_per_round(self):
        order=balanced_order(['a','b','c'],3)
        self.assertEqual(order,[['a','b','c'],['a','c','b'],['c','a','b']])
        self.assertTrue(all(sorted(r)==['a','b','c'] for r in order))


class RuntimeTests(unittest.TestCase):
    def test_dialects_and_no_shell(self):
        p=Policy('x',('CUDA0','ROCm0'),(3,2),999)
        b=bench_command(config(model='/tmp/a $(touch nope).gguf'),p,5)
        f=functional_command(config(),p)
        self.assertEqual(b[b.index('-dev')+1],'CUDA0/ROCm0')
        self.assertEqual(f[f.index('--device')+1],'CUDA0,ROCm0')
        self.assertIn('/tmp/a $(touch nope).gguf',b)
        self.assertEqual(b[b.index('-fitt')+1],'0')
        self.assertNotIn('-v',b)
    def test_enumeration_does_not_prove_execution(self):
        p=Policy('x',('CUDA0',),None,1)
        self.assertTrue(verify_placement(placement('CUDA0: NVIDIA GPU'),p,ModelLayout(3,(1,1),1,0)))
    def test_backend_sources_not_compute(self):
        text='node # 1 ( MUL_MAT): tensor (1M) [CPU weight] use=1,c=1: source (1M) [CUDA0 weight]'
        self.assertEqual(placement(text)['matrix_compute_backends'],['CPU'])
    def test_conflicting_layer_mapping(self):
        x=placement('load_tensors: layer 0 assigned to device CUDA0\nload_tensors: layer 0 assigned to device ROCm0')
        self.assertEqual(x['conflicting_layers'],['0'])
    def test_response_check(self):
        for answer in ('42',' 42\n','\t\r\n42 \t\r\n'):
            with self.subTest(answer=answer):self.assertTrue(response_correct(answer))
    def test_response_rejects_explanations_and_contradictions(self):
        for answer in ('42\nThis is incorrect','42\nI did not calculate it',
                       '42\nI was guessing','42\nNo, that is false','The answer is\n42'):
            with self.subTest(answer=answer):self.assertFalse(response_correct(answer))
    def test_response_rejects_repeated_wrong_or_empty_answers(self):
        for answer in ('',' \t\r\n','What is 6 times 7?\n','43','42\n43','42\n42',
                       '4242','4 2','+42','42.0','42!'):
            with self.subTest(answer=answer):self.assertFalse(response_correct(answer))
    def test_response_rejects_uncharacterized_control_markers(self):
        for answer in ('\x1b[32m42\x1b[0m','42\x00','42\b','42\v','\u00a042',
                       '42\n[END]','42<|eot_id|>','assistant\n42'):
            with self.subTest(answer=answer):self.assertFalse(response_correct(answer))


class AnalysisTests(unittest.TestCase):
    def test_integer_duration_recomputation(self):
        rows=benchmark_rows(config())
        rows[1]['avg_ts']=999999
        result=analysis.parse_benchmark(rows,config(),3)
        self.assertEqual(result['tg']['samples_tokens_s'][0],128)
        self.assertLess(result['tg']['mean'],128)
    def test_missing_or_wrong_workload_rejected(self):
        for field,value in [('n_depth',512),('type_k','q8_0'),('fit_target',2048),('samples_ns',[0,0,0])]:
            rows=benchmark_rows(config());rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):analysis.parse_benchmark(rows,config(),3)
    def test_nonfinite_statistics(self):
        with self.assertRaises(ValueError):analysis.describe([math.nan])
    def records(self):
        return [{'run_id':f'{name}-{r}','policy':{'name':name},'status':'ok','phase':'final','comparison_key':'same',
                 'validation_key':'passed','round':r,'metrics':{'tg':{'mean':v},'pp':{'mean':1000/v}}}
                for name,v in [('cuda-only',100),('rocm-only',70),('mixed-default',80),('mixed-3-2',90)] for r in range(3)]
    def test_fastest_single_gpu_wins(self):
        s=analysis.selection(self.records())
        self.assertEqual(s['selected'],'cuda-only')
        self.assertEqual(s['strongest_baseline'],'cuda-only')
        self.assertEqual(s['comparisons']['mixed-default']['speedup'],1.25)
    def test_launches_not_pseudoreplicates(self):
        r=self.records()[:1]
        self.assertIsNone(analysis.selection(r)['selected'])
    def test_failed_records_excluded(self):
        r=self.records();r[0]['status']='invalid'
        self.assertEqual(analysis.selection(r)['selected'],'mixed-3-2')
    def test_mismatched_experiments_refused(self):
        r=self.records();r[-1]['comparison_key']='other-model'
        with self.assertRaises(ValueError):analysis.selection(r)
    def test_duplicate_launch_refused(self):
        r=self.records()
        with self.assertRaises(ValueError):analysis.selection(r+r[:1])
    def test_policy_name_collision_refused(self):
        r=self.records();r[0]['policy']['gpu_layers']=22
        with self.assertRaises(ValueError):analysis.selection(r)
    def test_missing_validation_refused(self):
        r=self.records();r[0]['validation_key']=None
        with self.assertRaises(ValueError):analysis.selection(r)
    def test_finalist_baselines_preserved(self):
        r=self.records()
        for x in r:x['phase']='explore'
        self.assertEqual(analysis.finalists(r),['cuda-only','mixed-3-2','mixed-default','rocm-only'])


class CLITests(unittest.TestCase):
    def test_all_inference_commands_default_to_preview(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'snapshot.json';p.write_text(json.dumps(snapshot()))
            with patch('heterollm.cli.discover',side_effect=AssertionError('live query')), patch('heterollm.cli.Session',side_effect=AssertionError('execution')):
                for name in ('tune','validate','benchmark'):
                    out=io.StringIO()
                    with redirect_stdout(out):code=main([name,'--config',str(ROOT/'configs/llama33-70b.json'),'--snapshot',str(p)])
                    self.assertEqual(code,0)
                    self.assertTrue(json.loads(out.getvalue())['dry_run'])
    def test_snapshot_forbids_execution(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'snapshot.json';p.write_text(json.dumps(snapshot()))
            with redirect_stderr(io.StringIO()):
                code=main(['tune','--config',str(ROOT/'configs/llama33-70b.json'),'--snapshot',str(p),'--execute'])
            self.assertEqual(code,2)
