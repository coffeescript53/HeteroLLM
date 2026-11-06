"""CPU-only shared recorder/event pool and strict cross-trace acceptance."""
import copy,importlib.util,json,os,subprocess,tempfile,unittest
from pathlib import Path
from helpers import ROOT
from heterollm.timing_analysis import analyze_timing

class TimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='c2-timing-cpu-');cls.root=Path(cls.temp.name)
        cls.exe=cls.root/'fake'
        subprocess.run(['c++','-std=c++17','-O1','-pthread','-DHETEROLLM_CPU_TRACE=1',
            '-include','algorithm','-I',str(ROOT/'instrumentation/v3'),'-I',str(ROOT/'instrumentation/v2'),
            '-I',str(ROOT/'llama.cpp/ggml/include'),str(ROOT/'tests/fixtures/c2_timing_fake.cpp'),'-o',str(cls.exe)],check=True,capture_output=True,text=True,timeout=40)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def run_fake(self,mode='valid',enabled=True):
        d=Path(tempfile.mkdtemp(dir=self.root));env={**os.environ}
        env.pop('HETEROLLM_TRACE_FILE',None);env.pop('HETEROLLM_TIMING_FILE',None)
        if enabled:env.update(HETEROLLM_TRACE_FILE=str(d/'cpu.jsonl'),HETEROLLM_TIMING_FILE=str(d/'device.jsonl'))
        r=subprocess.run([str(self.exe),mode],env=env,capture_output=True,text=True,timeout=5)
        self.assertEqual(r.returncode,0,r.stderr)
        return r,d
    def valid(self):
        r,d=self.run_fake();return (d/'device.jsonl').read_text(),(d/'cpu.jsonl').read_text(),r.stderr
    def test_completed_durations_phase_join_data_and_callback_order(self):
        r,d=self.run_fake();s=analyze_timing((d/'device.jsonl').read_text(),(d/'cpu.jsonl').read_text(),r.stderr)
        self.assertEqual(s['device_records'],16);self.assertEqual(s['matched_host_stage_legs'],8)
        self.assertIsNone(s['cross_device_overlap']);self.assertIsNone(s['gpu_kernel_only_time'])
        disabled,path=self.run_fake(enabled=False);self.assertEqual(r.stdout,disabled.stdout)
        self.assertEqual(list(path.iterdir()),[])
    def test_complete_zero_host_stage_is_valid(self):
        r,d=self.run_fake('zero');s=analyze_timing((d/'device.jsonl').read_text(),(d/'cpu.jsonl').read_text(),r.stderr)
        self.assertEqual(s['matched_host_stage_legs'],0)
    def test_runtime_errors_and_incomplete_events_fail(self):
        for mode in ('not_ready','overflow','record_overflow','create_failure','record_failure','query_failure','elapsed_failure','nan','destroy_failure','abrupt','write_failure'):
            with self.subTest(mode=mode):
                r,d=self.run_fake(mode)
                with self.assertRaises(ValueError):analyze_timing((d/'device.jsonl').read_text(),(d/'cpu.jsonl').read_text(),r.stderr)
    def test_missing_truncated_malformed_or_error_trace_fails(self):
        t,c,e=self.valid()
        for changed,stderr in [('',e),(t[:len(t)//2],e),(t[:-2],e),('{bad}',e),(t,'HETEROLLM_TIMING_ERROR: write failure'),(t.replace('"id":0','"id":0,"id":0'),e)]:
            with self.assertRaises((ValueError,KeyError)):analyze_timing(changed,c,stderr)
        with self.assertRaises(ValueError):analyze_timing(t,c[:len(c)//2],e)
    def test_footer_counts_loss_and_unfinished_rejected(self):
        t,c,e=self.valid();rows=list(map(json.loads,t.splitlines()))
        for k,v in [('complete',False),('invalid',True),('pending',1),('started',17),('phases',3),('active_phase',4)]:
            r=copy.deepcopy(rows);r[-1][k]=v
            with self.assertRaises(ValueError):analyze_timing('\n'.join(map(json.dumps,r)),c,e)
    def test_wrong_phase_clock_ids_duration_or_missing_graph_rejected(self):
        t,c,e=self.valid();rows=list(map(json.loads,t.splitlines()));idx=next(i for i,r in enumerate(rows)if r['type']=='device')
        for k,v in [('phase',2),('id',False),('backend','ROCm1'),('start_after',0),('elapsed_ms',-1),('elapsed_ms',float('inf')),('graph',999),('thread',0),('stream','NULL')]:
            r=copy.deepcopy(rows);r[idx][k]=v
            with self.subTest(k=k,v=v),self.assertRaises(ValueError):analyze_timing('\n'.join(map(json.dumps,r)),c,e)
        r=copy.deepcopy(rows);r[0]['clock']='gpu_clock'
        with self.assertRaises(ValueError):analyze_timing('\n'.join(map(json.dumps,r)),c,e)
    def test_host_stage_bytes_and_leg_coverage_rejected(self):
        t,c,e=self.valid();rows=list(map(json.loads,t.splitlines()));idx=next(i for i,r in enumerate(rows)if r.get('kind')=='copy_d2h')
        for field,value in [('logical_bytes',17),('kind','copy_h2d')]:
            r=copy.deepcopy(rows);r[idx][field]=value
            with self.assertRaises(ValueError):analyze_timing('\n'.join(map(json.dumps,r)),c,e)

if __name__=='__main__':unittest.main()
