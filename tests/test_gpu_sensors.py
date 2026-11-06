from helpers import thermal_fixture
"""CPU fixtures only: temporary sysfs, mocked device open, synthetic child."""
from dataclasses import replace
from functools import partial
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from heterollm.gpu_sensors import SensorUnavailable, amd_read, amd_ready, gpu_sample, resume_render_node
from heterollm.hardware import clean_environment
from heterollm.pilot import bounded_gpu_readiness, fast_sample, maps_sample
from heterollm.supervisor import Limits, Reader, append_json, supervise

PCI = '0000:03:00.0'
DEVICES = {'CUDA0': '0000:05:00.0', 'ROCm0': PCI}


class SensorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dev = self.root/PCI
        for name, value in {'vendor':'0x1002','device':'0x7551','power/runtime_status':'active',
                            'power/control':'auto','power/autosuspend_delay_ms':'5000',
                            'mem_info_vram_total':str(32*1024**3),'mem_info_vram_used':'1024',
                            'gpu_busy_percent':'0','hwmon/hwmon1/temp1_input':'31000',
                            'hwmon/hwmon1/temp2_input':'33000'}.items():
            self.put(name, value)
        self.events = []

    def put(self, name, value):
        p = self.dev/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)

    def record(self, event, data):
        self.events.append({'event': event, 'data': data})

    def ready(self):
        return amd_ready(PCI, self.record, self.root, self.root/'nodes')

    def wake(self, *args):
        self.put('power/runtime_status','active')

    def test_active_to_suspended_then_fresh_recovery(self):
        with patch('heterollm.gpu_sensors.resume_render_node', side_effect=self.wake) as wake:
            self.assertEqual(self.ready()['temperature_c'],33)
            wake.assert_not_called()
            self.put('power/runtime_status','suspended')
            self.assertIsNone(amd_read(PCI,self.root)['temperature_c'])
            self.put('hwmon/hwmon1/temp2_input','41000')
            result = self.ready()
            self.assertEqual(result['temperature_c'],41)  # not the old active value
            self.assertEqual(result['readiness_refresh']['before']['runtime_status'],'suspended')
            wake.assert_called_once()
        self.assertEqual((self.dev/'power/control').read_text(),'auto')
        self.assertEqual((self.dev/'power/autosuspend_delay_ms').read_text(),'5000')

    def test_transition_during_temperature_read_recovers(self):
        before = amd_read(PCI,self.root)
        before.update(temperature_c=None,runtime_status_after='suspended')
        after = amd_read(PCI,self.root)
        with patch('heterollm.gpu_sensors.amd_read',side_effect=[before,after]), \
             patch('heterollm.gpu_sensors.resume_render_node') as wake:
            self.assertEqual(self.ready()['temperature_c'],33)
            wake.assert_called_once()

    def test_active_missing_temperature_fails_without_refresh(self):
        self.put('hwmon/hwmon1/temp2_input','invalid')
        with patch('heterollm.gpu_sensors.resume_render_node') as wake:
            with self.assertRaises(SensorUnavailable) as caught:self.ready()
            wake.assert_not_called()
        self.assertEqual(caught.exception.evidence['missing_fields'],['temperature_c'])
        self.assertIn('invalid',str(caught.exception))
        self.assertIn(PCI,str(caught.exception))

    def test_resume_does_not_replace_missing_vram_or_temperature(self):
        for path in ('mem_info_vram_used','hwmon/hwmon1/temp1_input'):
            with self.subTest(path=path):
                self.put('power/runtime_status','suspended'); self.put(path,'bad')
                with patch('heterollm.gpu_sensors.resume_render_node',side_effect=self.wake) as wake:
                    with self.assertRaises(SensorUnavailable) as caught:self.ready()
                    wake.assert_called_once()
                self.assertIn('after',caught.exception.evidence)
                self.put(path,'1024' if 'vram' in path else '31000')

    def test_failed_resume_retains_errno_and_raw_state(self):
        self.put('power/runtime_status','suspended')
        with patch('heterollm.gpu_sensors.resume_render_node',side_effect=PermissionError(13,'denied')):
            with self.assertRaises(SensorUnavailable) as caught:self.ready()
        e = caught.exception.evidence
        self.assertEqual(e['errno'],13)
        self.assertEqual(e['before']['raw']['power/control']['value'],'auto')
        self.assertEqual(e['refresh_attempts'],1)

    def test_raw_temperature_read_error_is_preserved(self):
        original=Path.read_text
        def read(path,*args,**kwargs):
            if path.name=='temp2_input':raise OSError(5,'synthetic sensor I/O error')
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',read):
            with self.assertRaises(SensorUnavailable) as caught:self.ready()
        raw=caught.exception.evidence['before']['raw']['temperatures']
        self.assertEqual(raw[str(self.dev/'hwmon/hwmon1/temp2_input')]['errno'],5)
        self.assertEqual(caught.exception.evidence['missing_fields'],['temperature_c'])

    def test_preflight_reader_cleanup_failure_blocks_admission(self):
        with patch('heterollm.pilot.Reader') as reader:
            reader.return_value.poll.return_value={'data':{}}
            reader.return_value.close.return_value=False
            with self.assertRaisesRegex(RuntimeError,'cleanup incomplete'):
                bounded_gpu_readiness(SimpleNamespace(devices=DEVICES,output=self.root))
            self.assertEqual(reader.call_args.args[-1],2)

    def test_transition_never_recovers_is_bounded_one_open(self):
        self.put('power/runtime_status','resuming')
        with patch('heterollm.gpu_sensors.resume_render_node') as wake, \
             patch('heterollm.gpu_sensors.time.sleep'), \
             patch('heterollm.gpu_sensors.stamp',return_value={}), \
             patch('heterollm.gpu_sensors.time.monotonic',side_effect=[0,1]):
            with self.assertRaises(SensorUnavailable):self.ready()
            wake.assert_called_once()

    def test_hot_sensor_and_invalid_vram_fail_without_refresh(self):
        for path,value in [('hwmon/hwmon1/temp2_input','85000'),('mem_info_vram_used',str(40*1024**3))]:
            original=(self.dev/path).read_text();self.put(path,value)
            with patch('heterollm.gpu_sensors.resume_render_node') as wake:
                with self.assertRaises(SensorUnavailable):self.ready()
                wake.assert_not_called()
            self.put(path,original)

    def test_wrong_amd_identity_never_opens(self):
        self.put('device','0x9999')
        with patch('heterollm.gpu_sensors.resume_render_node') as wake:
            with self.assertRaises(SensorUnavailable):self.ready()
            wake.assert_not_called()

    def render_fixture(self):
        (self.dev/'driver').symlink_to(self.root/'amdgpu')
        node=self.dev/'drm/renderD129';node.mkdir(parents=True)
        (node/'dev').write_text('226:129')
        (node/'device').symlink_to(self.dev)
        return SimpleNamespace(st_mode=stat.S_IFCHR|0o660,st_rdev=os.makedev(226,129))

    def test_verified_render_open_has_no_writes_or_ioctls(self):
        s=self.render_fixture()
        original=Path.stat
        def node_stat(path,*args,**kwargs):
            return s if path.parent==self.root/'nodes' else original(path,*args,**kwargs)
        with patch.object(Path,'stat',node_stat),patch('os.open',return_value=7) as opened, \
             patch('os.fstat',return_value=s),patch('os.close') as closed:
            resume_render_node(PCI,self.record,self.root,self.root/'nodes')
        opened.assert_called_once_with(self.root/'nodes/renderD129',os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW)
        closed.assert_called_once_with(7)
        self.assertEqual([x['event'] for x in self.events],['resume_open_intent','resume_open_complete'])

    def test_wrong_character_device_refused_before_open(self):
        self.render_fixture()
        s=SimpleNamespace(st_mode=stat.S_IFCHR|0o660,st_rdev=os.makedev(226,128))
        original=Path.stat
        def node_stat(path,*args,**kwargs):
            return s if path.parent==self.root/'nodes' else original(path,*args,**kwargs)
        with patch.object(Path,'stat',node_stat),patch('os.open') as opened:
            with self.assertRaises(ValueError):resume_render_node(PCI,self.record,self.root,self.root/'nodes')
            opened.assert_not_called()

    def test_nvidia_missing_fields_preserve_query(self):
        raw={'exit_code':0,'stdout':'00000000:05:00.0, RTX 5090, 32000, 32768, 768, N/A, driver, 0\n','stderr':''}
        with patch('heterollm.gpu_sensors.query',return_value=raw),patch('heterollm.gpu_sensors.amd_ready') as amd:
            with self.assertRaises(SensorUnavailable) as caught:gpu_sample(DEVICES)
            amd.assert_not_called()
        self.assertEqual(caught.exception.evidence['missing_fields'],['temperature_c'])
        self.assertEqual(caught.exception.evidence['raw'],raw)

    def test_reader_preserves_structured_failure(self):
        self.put('hwmon/hwmon1/temp1_input','bad')
        reader=Reader('gpu',self.ready,self.root,None,2)
        try:
            with self.assertRaises(RuntimeError):
                while reader.poll() is None:time.sleep(.01)
        finally:self.assertTrue(reader.close())
        saved=json.loads((self.root/'gpu-latest.json').read_text())
        self.assertEqual(saved['data']['missing_fields'],['temperature_c'])
        self.assertEqual(saved['data']['pci'],PCI)
        self.assertIn('bad',(self.root/'gpu.jsonl').read_text())

    def test_hung_refresh_is_killed_and_intent_survives(self):
        self.put('power/runtime_status','suspended')
        def record(event,data):append_json(self.root/'raw.jsonl',{'event':event,'data':data})
        def hang(pci,record,*args):
            record('resume_open_intent',{'pci':pci});time.sleep(10)
        with patch('heterollm.gpu_sensors.resume_render_node',side_effect=hang):
            reader=Reader('gpu',partial(amd_ready,PCI,record,self.root),self.root,None,.15)
            started=time.monotonic()
            try:
                with self.assertRaisesRegex(RuntimeError,'timeout'):
                    while reader.poll() is None:time.sleep(.01)
            finally:self.assertTrue(reader.close())
        self.assertLess(time.monotonic()-started,1)
        self.assertIn('resume_open_intent',(self.root/'raw.jsonl').read_text())
        self.assertIn('suspended',(self.root/'raw.jsonl').read_text())

    def test_final_admission_transition_recovery_or_no_exec(self):
        for recover in (True,False):
            self.put('power/runtime_status','suspended')  # after active preflight / idle window
            def fast(pid):
                s=fast_sample(pid);s['memory']['MemAvailable']=64*1024**3
                s['memory']['HardwareCorrupted']=0;s['swap_used']=0;return s
            def gpu():
                return {'CUDA0':{'temperature_c':30},'ROCm0':self.ready()}
            out=self.root/str(recover)
            action=self.wake if recover else PermissionError(13,'no device access')
            with patch('heterollm.gpu_sensors.resume_render_node',side_effect=action):
                r=supervise([sys.executable,'-c','print("42")'],out,clean_environment(),lambda:{'swap_baseline':0},
                    lambda pid:{'thermal':thermal_fixture,'fast':partial(fast,pid),'gpu':gpu,'maps':partial(maps_sample,pid,'/synthetic')},
                    lambda o,e:{'errors':[] if o.strip()=='42' else ['bad']},replace(Limits(),overall=4,term_grace=.05))
            self.assertTrue(r['cleanup_complete']);self.assertTrue(r['reader_cleanup_complete'])
            self.assertEqual(r['launches'],int(recover),r)
            self.assertEqual('exec_release' in (out/'events.jsonl').read_text(),recover)


if __name__ == '__main__':unittest.main()
