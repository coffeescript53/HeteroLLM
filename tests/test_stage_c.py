from helpers import thermal_fixture
"""CPU fixtures only. Never invoke a GPU runtime or read the real model payload."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

from helpers import benchmark_rows, config, snapshot
from heterollm import stage_c as c0
from heterollm.analysis import parse_benchmark
from heterollm.gguf import ModelLayout
from heterollm.pilot import fast_sample, maps_sample
from heterollm.planner import estimate
from heterollm.supervisor import Limits, admission_reason, guard
from test_supervisor import gpu_fixture

GIB = 1024**3
LAYOUT = ModelLayout(37*GIB, (450*1024**2,)*81, 600*1024**2, 4096)


def hardware():
    value = snapshot()
    value['source']['exit_code'] = 0
    value['telemetry']['memory'].update(MemTotal=98774167552, MemAvailable=64*GIB)
    for group in ('nvidia', 'amd'):
        value['telemetry'][group][0]['used_bytes'] = 1024
    return value


def rows(conf, policy):
    value = benchmark_rows(conf, 1)
    for row in value:
        row.update(load_mode='none', model_filename=conf.model, devices='/'.join(policy.devices),
                   n_gpu_layers=policy.gpu_layers, cpu_mask='0x0', cpu_strict=False,
                   poll=50, main_gpu=0, build_commit=conf.source_revision[:10],
                   tensor_split='1.00/1.00' if policy.tensor_split else '0.00')
    return value


def evidence(conf, policy, cuda_count=None):
    screen = estimate(policy, LAYOUT, hardware(), conf).to_dict()
    mapping = {i: d for d, ids in screen['estimated_layers'].items() for i in ids}
    if cuda_count is not None:
        mapping = {i: 'CUDA0' if i < cuda_count else 'ROCm0' for i in range(81)}
    text = 'load_tensors: loading model tensors, this can take a while... (load_mode = none)\n'
    text += '\n'.join(f'load_tensors: layer {i} assigned to device {mapping.get(i, "CPU")}' for i in range(81))
    text += '\nn_ctx = 512\nn_ctx = 256\n'
    for d in policy.devices:
        text += (f'using device {d} (fixture GPU) ({conf.devices[d]})\n'
                 f'{d} model buffer size = 20000 MiB\n{d} KV buffer size = 80 MiB\n'
                 f'{d} compute buffer size = 100 MiB\nnode # 1 (MUL_MAT): op [{d} buffer] c=1: result\n')
    text += 'CPU_REPACK model buffer size = 10000 MiB\n'
    if len(policy.devices) == 1:
        text += 'node # 1 (MUL_MAT): op [CPU buffer] c=1: result\n'
    return text, screen


class StageCContractTests(unittest.TestCase):
    def test_exact_four_commands_and_controls(self):
        conf = config()
        self.assertEqual([(p.name, p.gpu_layers) for p in c0.POLICIES],
                         [('mixed-equal',999),('mixed-default',999),('cuda-only',56),('rocm-only',59)])
        controls = {'-lm':'none','-lzm':'off','-p':'512','-n':'128','-d':'0','-r':'1',
                    '-b':'512','-ub':'128','-t':'16','-fitt':'0','-fitc':'0','-fa':'off',
                    '-ctk':'f16','-ctv':'f16','-nkvo':'0','-nopo':'0','--no-host':'0','--repack':'1',
                    '-embd':'0','-ncmoe':'0','-mg':'0','--prio':'0','--delay':'0',
                    '--poll':'50','--cpu-strict':'0','-C':'0x0'}
        for policy in c0.POLICIES:
            argv = c0.command(conf, policy)
            for flag, expected in controls.items():
                self.assertEqual(argv.count(flag), 1)
                self.assertEqual(argv[argv.index(flag)+1], expected)
            self.assertIn('--offline', argv); self.assertIn('-v', argv)
            self.assertEqual('-ts' in argv, policy.name == 'mixed-equal')
            self.assertEqual(argv[argv.index('-dev')+1], '/'.join(policy.devices))
        self.assertEqual(c0.environment(conf)['GGML_SCHED_DEBUG'], '2')

    def test_plan_preview_never_discovers_or_executes(self):
        with patch.object(c0,'discover',side_effect=AssertionError('GPU discovery')), \
             patch.object(c0,'supervise',side_effect=AssertionError('execution')), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(c0.main(['--plan',str(c0.ROOT/'configs/stage-c0-plan.json')]),0)
        value=json.loads(out.getvalue())
        self.assertEqual(len(value['launches']),4)
        self.assertFalse(value['timings_eligible_for_comparison'])
        self.assertEqual(value['output_exists'], Path(value['output']).exists())

    def test_unreviewed_plan_controls_are_refused(self):
        plan_path=c0.ROOT/'configs/stage-c0-plan.json'
        original=json.loads(plan_path.read_text())
        conf=json.loads(Path(original['config']).read_text())
        for change in ({'max_launches':16},{'suite_seconds':14400},{'max_swap_growth_mib':128},
                       {'threads':8},{'gpu_reserve_mib':1024},{'minimum_available_ram_mib':1024}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                p=Path(temp)
                (p/'config.json').write_text(json.dumps({**conf,**change}))
                (p/'plan.json').write_text(json.dumps({**original,'config':str(p/'config.json')}))
                with self.assertRaises(ValueError): c0.load_plan(p/'plan.json')
        for stage in ('C1','C','B'):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'plan.json';p.write_text(json.dumps({**original,'stage':stage}))
                with self.assertRaises(ValueError): c0.load_plan(p)

    def test_limits_change_only_reviewed_admission_and_rss(self):
        old=asdict(Limits());new=asdict(c0.C0_LIMITS)
        changes={k:new[k] for k in new if new[k]!=old[k]}
        self.assertEqual(changes,{'rss_cap':32*GIB,'admission_ram':40*GIB,'admission_temperature':45})
        self.assertEqual(old['rss_cap'],8*GIB)
        self.assertEqual(new['swap_growth'],256*1024**2)
        self.assertEqual(new['temperature'],85)
        self.assertEqual(new['ram_floor'],4*GIB)
        self.assertEqual(new['overall'],900)

    def test_valid_rows_and_legacy_mmap_contract(self):
        for p in c0.POLICIES:
            value=rows(config(),p)
            self.assertEqual(len(c0.validate_rows(json.dumps(value),config(),p)),2)
            with self.assertRaises(ValueError):parse_benchmark(value,config(),1)
        parse_benchmark(benchmark_rows(config(),1),config(),1)

    def test_wrong_identity_workload_mode_or_repetition_is_rejected(self):
        conf=config();policy=c0.POLICIES[0]
        invalid={'load_mode':'mmap','lazy_mode':'on','n_prompt':511,'n_gen':128,'n_depth':1,
                 'n_batch':256,'n_threads':8,'fit_target':1,'fit_min_ctx':1,'model_filename':'other.gguf',
                 'devices':'CUDA0','n_gpu_layers':56,'tensor_split':'2/1','build_commit':'abcdef0',
                 'samples_ns':[1,2,3],'flash_attn':False,'no_kv_offload':0,'poll':True,'cpu_mask':'0x1',
                 'samples_ts':[float('nan')]}
        for key,value in invalid.items():
            with self.subTest(key=key):
                data=rows(conf,policy);data[0][key]=value
                with self.assertRaises((ValueError,TypeError)):
                    c0.validate_rows(json.dumps(data),conf,policy)
        for mode in ('dio','unsupported','none)'):
            data=rows(conf,policy);data[0]['load_mode']=mode
            with self.assertRaises(ValueError):c0.validate_rows(json.dumps(data),conf,policy)

    def test_duplicate_fields_nonfinite_or_truncated_json_refused(self):
        for text in ('[{"x":1,"x":2}]','[NaN]','[Infinity]','[{"x":1}', '{}'):
            with self.subTest(text=text),self.assertRaises(ValueError):
                c0.validate_rows(text,config(),c0.POLICIES[0])

    def validation(self, policy, transform=lambda s:s, cuda_count=None):
        with tempfile.TemporaryDirectory() as temp,patch.object(c0,'read_layout',return_value=LAYOUT):
            conf=config(output=temp)
            stderr,screen=evidence(conf,policy,cuda_count)
            (Path(temp)/'preflight.json').write_text(json.dumps({'screening':screen}))
            return c0.validate_diagnostic(conf,policy,json.dumps(rows(conf,policy)),transform(stderr))

    def test_complete_placement_all_four_policies(self):
        for p in c0.POLICIES:
            with self.subTest(policy=p.name):
                result=self.validation(p)
                self.assertEqual(result['errors'],[])
                self.assertEqual(len(result['placement']['layers']),81)
                self.assertFalse(result['timings_eligible_for_comparison'])
                self.assertFalse(result['semantic_answer_test'])
                self.assertTrue(result['placement']['all_compute_buffer_lines'])

    def test_missing_selected_backend_kv_allocation_fails(self):
        self.assertTrue(self.validation(c0.POLICIES[0],
            lambda s:s.replace('ROCm0 KV buffer size = 80 MiB',''))['errors'])

    def test_missing_layer_cpu_compute_or_host_allocation_rejected(self):
        transforms=(lambda s:s.replace('layer 1 assigned','layer 0 assigned'),
                    lambda s:s.replace('[CPU buffer] c=1:','[CPU buffer] c=0:'),
                    lambda s:s.replace('CPU_REPACK model buffer size = 10000 MiB',''),
                    lambda s:s.replace('n_ctx = 256','n_ctx = 128'),
                    lambda s:s.replace('0000:05:00.0','0000:06:00.0'),
                    lambda s:s+'\nusing device ROCm1 (APU) (0000:10:00.0)\n')
        for transform in transforms:
            self.assertTrue(self.validation(c0.POLICIES[2],transform)['errors'])

    def test_default_boundary_observed_equal_boundary_enforced(self):
        changed=self.validation(c0.POLICIES[1],cuda_count=42)
        self.assertEqual(changed['errors'],[])
        self.assertFalse(changed['placement']['matches_preflight_prediction'])
        self.assertTrue(self.validation(c0.POLICIES[0],cuda_count=42)['errors'])

    def test_mixed_comparison_same_and_different_allocations(self):
        equal=self.validation(c0.POLICIES[0])
        default=self.validation(c0.POLICIES[1],cuda_count=41)
        def result(e,d):
            return c0.mixed_comparison([{'policy':n,'result':{'status':'passed','validation':v}}
                for n,v in [('mixed-equal',e),('mixed-default',d)]])
        self.assertTrue(result(equal,default)['same_layer_mapping'])
        default=self.validation(c0.POLICIES[1],cuda_count=40)
        self.assertEqual(result(equal,default)['changed_layers'],['40'])


class StageCAdmissionTests(unittest.TestCase):
    def test_40_gib_ram_and_45_degree_launch_admission(self):
        fast={'memory':{'MemAvailable':40*GIB}};gpu=gpu_fixture()
        self.assertIsNone(admission_reason(fast,gpu,{},c0.C0_LIMITS))
        fast['memory']['MemAvailable']-=1
        self.assertEqual(admission_reason(fast,gpu,{},c0.C0_LIMITS),'admission_ram_changed')
        fast['memory']['MemAvailable']+=1;gpu['ROCm0']['temperature_c']=46
        self.assertEqual(admission_reason(fast,gpu,{},c0.C0_LIMITS),'admission_temperature')

    def test_runtime_capacity_ceiling_and_fresh_sensor_growth(self):
        fast={'memory':{'MemAvailable':40*GIB}};gpu=gpu_fixture()
        pre={'screening':{'gpu_estimated_bytes':{'CUDA0':27*GIB}},
             'capacity_baseline':{'CUDA0':{'runtime_free_bytes':30*GIB,'sensor_used_bytes':1024}}}
        self.assertIsNone(admission_reason(fast,gpu,pre,c0.C0_LIMITS))
        gpu['CUDA0']['used_bytes']+=1
        self.assertEqual(admission_reason(fast,gpu,pre,c0.C0_LIMITS),'admission_vram_changed')
        gpu['CUDA0']['used_bytes']=0
        pre['capacity_baseline']['CUDA0']['runtime_free_bytes']-=1
        self.assertEqual(admission_reason(fast,gpu,pre,c0.C0_LIMITS),'admission_vram_changed')
        pre['capacity_baseline']['CUDA0']['runtime_free_bytes']=31*GIB
        gpu['CUDA0']['free_bytes']=29*GIB
        self.assertEqual(admission_reason(fast,gpu,pre,c0.C0_LIMITS),'admission_vram_changed')

    def test_all_runtime_safety_thresholds_remain_active(self):
        base={'memory':{'MemAvailable':40*GIB,'HardwareCorrupted':0},'swap_used':0,
              'process':{'VmRSS':30*GIB,'VmHWM':32*GIB}}
        self.assertIsNone(guard(base,gpu_fixture(),{'model_mappings':[]},0,c0.C0_LIMITS))
        for target,key,value,expected in (
            ('memory','MemAvailable',4*GIB-1,'host_memory_pressure'),
            ('memory','HardwareCorrupted',1,'hardware_memory_error'),
            ('process','VmHWM',32*GIB+1,'process_rss_limit')):
            data=deepcopy(base);data[target][key]=value
            self.assertEqual(guard(data,gpu_fixture(),{'model_mappings':[]},0,c0.C0_LIMITS),expected)
        data=deepcopy(base);data['swap_used']=256*1024**2+1
        self.assertEqual(guard(data,gpu_fixture(),{'model_mappings':[]},0,c0.C0_LIMITS),'swap_growth')
        gpu=gpu_fixture();gpu['ROCm0']['temperature_c']=85
        self.assertEqual(guard(base,gpu,{'model_mappings':[]},0,c0.C0_LIMITS),'temperature_limit')
        self.assertEqual(guard(base,gpu_fixture(),{'model_mappings':['model']},0,c0.C0_LIMITS),'unexpected_model_mapping')

    def preflight_case(self, policy, mutate=lambda s:s, error=None):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            directory=Path(temp);conf=config(output=temp);snap=hardware()
            mutate(snap)
            (directory/'runtime.json').write_text('{}')
            (directory/'deps.json').write_text('{"files":{}}')
            (directory/'baseline.json').write_text(json.dumps({'signature':c0.hardware_signature(hardware()),'controls':{}}))
            plan={'receipt':'synthetic','runtime_baseline':str(directory/'runtime.json'),
                  'dependency_baseline':str(directory/'deps.json'),'hardware_baseline':str(directory/'baseline.json'),
                  'mem_total_bytes':98774167552}
            replacements={'load_receipt':{},'runtime_identity':{},'dependencies':{'files':{}},
                          'concurrent_runtimes':[],
                          'discover':snap,'host_controls':{},'read_layout':LAYOUT,
                          'fast_sample':{},'maps_sample':{},'bounded_gpu_readiness':gpu_fixture()}
            for name,value in replacements.items():stack.enter_context(patch.object(c0,name,return_value=value))
            stack.enter_context(patch.object(c0,'IDLE_SECONDS',0))
            stack.enter_context(patch('heterollm.stage_c.shutil.disk_usage',return_value=type('Disk',(),{'free':20*GIB})()))
            if error:
                with self.assertRaisesRegex(ValueError,error):c0.preflight(conf,plan,policy,123)
                self.assertTrue((directory/'hardware.json').exists())
                return {p.name:json.loads(p.read_text()) for p in directory.glob('*.json')}
            value=c0.preflight(conf,plan,policy,123)
            self.assertEqual(value['swap_baseline'],123)
            self.assertEqual(value['policy']['gpu_layers'],policy.gpu_layers)
            return value

    def test_preflight_fixed_policies_and_frozen_swap_baseline(self):
        for policy in c0.POLICIES:self.preflight_case(policy)

    def test_preflight_insufficient_ram_fails(self):
        self.preflight_case(c0.POLICIES[0],
                            lambda s:s['telemetry']['memory'].update(MemAvailable=40*GIB-1),'>=40 GiB')

    def test_single_gpu_capacity_failure_retains_exact_layer_request(self):
        for policy in c0.POLICIES[2:]:
            required=estimate(policy,LAYOUT,hardware(),config()).gpu_estimated_bytes[policy.devices[0]]+3*GIB
            def low(s):
                for device in s['devices']:
                    if device['name']==policy.devices[0]:device['free_bytes']=required-1
            files=self.preflight_case(policy,low,'fixed policy fails capacity')
            self.assertEqual(files['capacity-screen.json']['policy']['gpu_layers'],policy.gpu_layers)
            self.assertFalse(files['capacity-screen.json']['admissible'])

    def test_preflight_driver_or_source_drift_fails(self):
        self.preflight_case(c0.POLICIES[0],lambda s:s['source_status'].update(stdout=' M changed'),'pinned source')
        self.preflight_case(c0.POLICIES[0],lambda s:s['telemetry']['nvidia'][0].update(driver='other'),'identity changed')

    def test_only_known_runtime_process_names_are_collected(self):
        with tempfile.TemporaryDirectory() as temp:
            for i,name in enumerate(('llama-bench','llama-cli','private-process'),1):
                p=Path(temp)/str(i);p.mkdir();(p/'comm').write_text(name+'\n')
            self.assertEqual(sorted(c0.concurrent_runtimes(Path(temp)),key=lambda r:r['pid']),
                             [{'pid':1,'name':'llama-bench'},{'pid':2,'name':'llama-cli'}])


class StageCExecutionTests(unittest.TestCase):
    def execute_fixture(self, fail_at=None, interrupt=False, budget=None):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            conf=config(output=str(Path(temp)/'suite'));seen=[]
            def fake_supervise(argv,directory,env,pre,samplers,validate,limits,**kwargs):
                ordinal=len(seen)+1;seen.append((argv,pre.args[-1],limits))
                Path(directory).mkdir()
                stderr,screen=evidence(replace(conf,output=directory),c0.POLICIES[ordinal-1])
                (Path(directory)/'preflight.json').write_text(json.dumps({'screening':screen}))
                validation=validate(json.dumps(rows(conf,c0.POLICIES[ordinal-1])),stderr)
                if interrupt:os.kill(os.getpid(),signal.SIGTERM)
                return {'status':'failed' if ordinal==fail_at else 'passed','reason':'synthetic' if ordinal==fail_at else None,
                        'launches':1,'validation':validation,'cleanup_complete':True,'reader_cleanup_complete':True}
            stack.enter_context(patch.object(c0,'supervise',side_effect=fake_supervise))
            stack.enter_context(patch.object(c0,'read_layout',return_value=LAYOUT))
            stack.enter_context(patch.object(c0,'COOLDOWN_SECONDS',0))
            if budget is not None:stack.enter_context(patch.object(c0,'SUITE_SECONDS',budget))
            result=c0.execute(conf,{'test':'CPU fixture'})
            self.assertEqual(json.loads((Path(conf.output)/'suite-result.json').read_text()),result)
            self.assertTrue((Path(conf.output)/'suite-events.jsonl').exists())
            return result,seen

    def test_four_launch_order_independent_outputs_and_no_c1(self):
        result,seen=self.execute_fixture()
        self.assertEqual(result['status'],'passed',result)
        self.assertEqual(result['launches'],4)
        self.assertEqual([r['policy'] for r in result['attempts']],[p.name for p in c0.POLICIES])
        self.assertEqual(len({r['directory'] for r in result['attempts']}),4)
        self.assertEqual(len({s[1] for s in seen}),1)
        self.assertFalse(result['timings_eligible_for_comparison'])

    def test_failure_stops_at_each_position_without_retry(self):
        for ordinal in range(1,5):
            result,seen=self.execute_fixture(fail_at=ordinal)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(len(seen),ordinal)

    def test_interrupt_or_insufficient_suite_budget_stops(self):
        result,seen=self.execute_fixture(interrupt=True)
        self.assertEqual(result['status'],'failed');self.assertEqual(len(seen),1)
        self.assertEqual(result['received_signals'],[signal.SIGTERM])
        result,seen=self.execute_fixture(budget=899)
        self.assertEqual(result['status'],'failed');self.assertEqual(seen,[])

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):c0.execute(config(output=temp),{})

    def test_four_real_cpu_subprocesses_through_supervisor_and_validation(self):
        # This is the adapter's complete process path with Python children and
        # synthetic GPU/large-RAM counters. No llama executable can be invoked.
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            conf=config(output=str(Path(temp)/'suite'))
            def pre(child,plan,policy,baseline):
                return {'screening':evidence(child,policy)[1],'swap_baseline':baseline}
            def cmd(child,policy):
                stderr=evidence(child,policy)[0];stdout=json.dumps(rows(child,policy))
                return [sys.executable,'-c',
                        f'import sys,time;print({stdout!r},flush=True);print({stderr!r},file=sys.stderr,flush=True);time.sleep(.3)']
            def samplers(child,pid):
                def fast():
                    value=fast_sample(pid);value['memory']['MemAvailable']=64*GIB
                    value['memory']['HardwareCorrupted']=0;value['swap_used']=0
                    return value
                return {'thermal':thermal_fixture,'fast':fast,'gpu':gpu_fixture,'maps':lambda:maps_sample(pid,'/synthetic')}
            for name,value in [('preflight',pre),('command',cmd),('sampler_factory',samplers)]:
                stack.enter_context(patch.object(c0,name,side_effect=value))
            stack.enter_context(patch.object(c0,'read_layout',return_value=LAYOUT))
            stack.enter_context(patch.object(c0,'COOLDOWN_SECONDS',0))
            stack.enter_context(patch.object(c0,'C0_LIMITS',replace(c0.C0_LIMITS,overall=5,preflight=2,term_grace=.05)))
            result=c0.execute(conf,{'test':'Python children only'})
            self.assertEqual(result['status'],'passed',result)
            self.assertEqual(result['launches'],4)
            for attempt in result['attempts']:
                r=attempt['result'];path=Path(attempt['directory'])
                self.assertTrue(r['cleanup_complete']);self.assertTrue(r['reader_cleanup_complete'])
                self.assertGreater(r['peak_child_rss_bytes'],0)
                self.assertEqual(r['validation']['errors'],[])
                for name in ('stdout.txt','stderr.txt','wait4-rusage.json','resource-summary.json',
                             'fast.jsonl','gpu.jsonl','maps.jsonl','events.jsonl'):
                    self.assertTrue((path/name).exists(),name)
