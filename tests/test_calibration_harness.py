"""Execute the actual capture source against CPU fake API callbacks only."""
import json,os,subprocess,tempfile,unittest
from pathlib import Path
from helpers import ROOT
from heterollm import calibration_analysis as a
from heterollm.timing_analysis import analyze_timing

class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='heterollm-cpu-calibration-');cls.root=Path(cls.temp.name)
        for arm in ('U','T'):
            command=['c++','-std=c++17','-O1','-rdynamic','-I'+str(ROOT/'llama.cpp/include'),'-I'+str(ROOT/'llama.cpp/ggml/include'),
                str(ROOT/'instrumentation/calibration/equivalence.cpp'),str(ROOT/'tests/fixtures/calibration_fake_runtime.cpp'),'-ldl','-o',str(cls.root/arm)]
            if arm=='T':command.insert(1,'-DFAKE_TIMING')
            subprocess.run(command,check=True,capture_output=True,text=True,timeout=40)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def run_fake(self,arm,extra=None,fixture=None):
        out=Path(tempfile.mkdtemp(dir=self.root));env={**os.environ}
        for k in ('HETEROLLM_TRACE_FILE','HETEROLLM_TIMING_FILE'):env.pop(k,None)
        if arm=='T':env.update(HETEROLLM_TRACE_FILE=str(out/'fake-cpu'),HETEROLLM_TIMING_FILE=str(out/'fake-device'))
        env.update(extra or {})
        backend=ROOT/('llama.cpp' if arm=='U' else 'runtime-c2-phase-v3')/'build-mixed/bin'
        r=subprocess.run([str(self.root/arm),arm,'FAKE_NO_MODEL',str(fixture or ROOT/'instrumentation/calibration/tokens.txt'),str(out),str(backend)],env=env,capture_output=True,text=True,timeout=15)
        return r,out
    def test_actual_harness_callback_order_batch_tokens_sync_and_output_bytes(self):
        u,up=self.run_fake('U');t,tp=self.run_fake('T')
        self.assertEqual(u.returncode,0,u.stderr);self.assertEqual(t.returncode,0,t.stderr)
        self.assertEqual(u.stdout,t.stdout)
        self.assertEqual((up/'logits.f32').read_bytes(),(tp/'logits.f32').read_bytes())
        result=a.compare_captures(up,tp);self.assertTrue(result['passed']);self.assertEqual(result['compared_values'],16545024)
        self.assertEqual(result['different_values'],0)
    def test_decode_failure_nonfinite_and_wrong_fixture_leave_incomplete_capture(self):
        for failure in ('FAKE_DECODE_ERROR','FAKE_NONFINITE'):
            r,out=self.run_fake('T',{failure:'1'});self.assertNotEqual(r.returncode,0)
            with self.assertRaises((ValueError,FileNotFoundError)):a.validate_capture(out,'T')
        fixture=self.root/'bad-tokens';fixture.write_text('128000\n')
        r,out=self.run_fake('U',fixture=fixture);self.assertNotEqual(r.returncode,0)
    def test_wrong_runtime_environment_fails_before_model(self):
        r,out=self.run_fake('U',{'HETEROLLM_TRACE_FILE':'unexpected'});self.assertNotEqual(r.returncode,0)
        self.assertFalse((out/'logits.f32').exists())

class RepeatedRecorderTests(unittest.TestCase):
    def test_native_cpu_fake_recorder_accepts_actual_eight_phase_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);source=(ROOT/'tests/fixtures/c2_timing_fake.cpp').read_text()
            source=source.replace('const char *names[]={"warmup_pp","pp512","warmup_tg","tg128"};',
                'const char *names[]={"warmup_pp","pp512","pp512","pp512","warmup_tg","tg128","tg128","tg128"}; int ix=0; const int reps[]={0,0,1,2,0,0,1,2};')
            source=source.replace('ht3_phase_scope phase(name,0);','ht3_phase_scope phase(name,reps[ix++]);')
            source=source.replace('==16);','==32);').replace('"record")==32);','"record")==64);')
            (d/'fake.cpp').write_text(source)
            subprocess.run(['c++','-std=c++17','-O1','-pthread','-DHETEROLLM_CPU_TRACE=1','-include','algorithm',
                '-I'+str(ROOT/'instrumentation/v3'),'-I'+str(ROOT/'instrumentation/v2'),'-I'+str(ROOT/'llama.cpp/ggml/include'),str(d/'fake.cpp'),'-o',str(d/'fake')],check=True,capture_output=True,text=True,timeout=40)
            env={**os.environ,'HETEROLLM_TRACE_FILE':str(d/'cpu'),'HETEROLLM_TIMING_FILE':str(d/'timing')}
            r=subprocess.run([str(d/'fake')],env=env,capture_output=True,text=True,timeout=5)
            self.assertEqual(r.returncode,0,r.stderr)
            result=analyze_timing((d/'timing').read_text(),(d/'cpu').read_text(),r.stderr,repetitions=3)
            self.assertEqual(result['device_records'],32);self.assertEqual(result['matched_host_stage_legs'],16)
            env.pop('HETEROLLM_TRACE_FILE');env.pop('HETEROLLM_TIMING_FILE')
            disabled=subprocess.run([str(d/'fake')],env=env,capture_output=True,text=True,timeout=5)
            self.assertEqual(disabled.returncode,0,disabled.stderr);self.assertEqual(disabled.stdout,r.stdout)

if __name__=='__main__':unittest.main()
