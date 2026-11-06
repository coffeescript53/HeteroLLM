"""CPU-only sensor trees, deterministic time, and owned sleeping Python children."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helpers import thermal_fixture
from heterollm import host_thermal as ht
from heterollm.supervisor import evidence_sizes


def reading(t, cpu=40, dimms=(35,36)):
    r=thermal_fixture(cpu,dimms)
    r.update(started_monotonic=t,ended_monotonic=t)
    return r


class ThermalPolicyTests(unittest.TestCase):
    def test_normal_and_maxima_for_each_physical_sensor(self):
        g=ht.ThermalGuard()
        for t in range(40):self.assertIsNone(g.observe(reading(t,70,(42,44)),t))
        self.assertEqual(g.summary()['max_observed_c'],{'cpu':70,'dimm-0051':42,'dimm-0053':44})
        self.assertEqual(g.count,40)

    def test_brief_cpu_boost_at_tjmax_and_dimm_spikes_recover(self):
        g=ht.ThermalGuard()
        for t in range(50):
            self.assertIsNone(g.observe(reading(t,95 if 2<=t<=8 else 80,
                                                    (60.5,40) if 12<=t<=16 else (40,40)),t))
        self.assertFalse(g.high_since);self.assertFalse(g.warnings)

    def test_sustained_stop_and_hysteresis_do_not_chatter(self):
        for kind in ('cpu','dimm'):
            g=ht.ThermalGuard()
            for t in range(26):
                cpu=95 if kind=='cpu' else 40
                dimms=(40,60) if kind=='dimm' else (40,40)
                stop=g.observe(reading(t,cpu,dimms),t)
                self.assertEqual(stop,('host_thermal_sustained:'+('cpu' if kind=='cpu' else 'dimm-0053')) if t==25 else None)
            self.assertTrue(g.warnings)
        g=ht.ThermalGuard();g.observe(reading(0,95),0)
        for t in range(1,25):self.assertIsNone(g.observe(reading(t,94),t))
        self.assertEqual(g.observe(reading(25,94),25),'host_thermal_sustained:cpu')

    def test_critical_and_hardware_alarms_stop_without_dwell(self):
        for r,reason in [(reading(0,95.125),'critical:cpu'),
                         (reading(0,40,(65,40)),'critical:dimm-0051')]:
            self.assertEqual(ht.ThermalGuard().observe(r,0),'host_thermal_'+reason)
        for key in ht.SENSOR_IDS:
            r=reading(0);r['sensors'][key]['alarms']={'temp1_crit_alarm':1}
            self.assertEqual(ht.ThermalGuard().observe(r,0),'host_thermal_hardware_alarm:'+key)

    def test_missing_invalid_stale_replayed_and_identity_change_fail(self):
        mutations=[lambda r:r['sensors'].pop('dimm-0053'),
                   lambda r:r['sensors']['cpu'].update(temperature_c=float('nan')),
                   lambda r:r['sensors']['cpu'].update(temperature_c=True),
                   lambda r:r['sensors']['cpu'].update(temperature_c=-1),
                   lambda r:r['sensors']['cpu'].update(unit='F'),
                   lambda r:r.update(started_monotonic=-4),
                   lambda r:r.update(ended_monotonic=1),
                   lambda r:r['sensors']['cpu'].update(alarms={'bad':2})]
        for mutate in mutations:
            r=reading(0);mutate(r)
            with self.assertRaises((ValueError,KeyError)):ht.ThermalGuard().observe(r,0)
        for t in (0,4):
            g=ht.ThermalGuard();g.observe(reading(0),0)
            with self.assertRaisesRegex(ValueError,'stale'):g.observe(reading(t),t)
        g=ht.ThermalGuard();g.observe(reading(0),0)
        r=reading(1);r['sensors']['dimm-0051']['identity']='different DIMM'
        with self.assertRaisesRegex(ValueError,'identity'):g.observe(r,1)

    def test_admission_refuses_hot_sensors_and_missing_dimm(self):
        for r in (reading(0,90),):
            self.assertIn('admission',ht.ThermalGuard().observe(r,0,admission=True))
        r=reading(0);r['sensors'].pop('dimm-0051')
        with self.assertRaises(ValueError):ht.ThermalGuard().observe(r,0,admission=True)

    def test_persistence_is_per_dimm_not_alternating_hot_sticks(self):
        g=ht.ThermalGuard()
        for t in range(60):
            dimms=(60,40) if (t//10)%2==0 else (40,60)
            self.assertIsNone(g.observe(reading(t,40,dimms),t))


    def test_warning_only_and_sticky_high_alarm_never_start_stop_timer(self):
        g=ht.ThermalGuard();states=[]
        for t in range(40):
            r=reading(t,40,(55 if t<5 else 40,40))
            r['sensors']['dimm-0051']['alarms']={'temp1_max_alarm':1}
            self.assertIsNone(g.observe(r,t))
            states.extend(x['state'] for x in g.transitions)
        self.assertEqual(states,['warning'])
        self.assertFalse(g.high_since)
        self.assertEqual(g.summary()['high_alarm_warnings']['dimm-0051']['asserted_samples'],40)
        self.assertTrue(g.summary()['high_alarm_warnings']['dimm-0051']['may_be_sticky'])
        self.assertIsNone(g.observe(reading(40),40))
        self.assertEqual(g.transitions[0]['state'],'recovered')

    def test_cpu_high_alarm_and_dimm_critical_other_faults_remain_immediate(self):
        for key,name in [('cpu','temp1_max_alarm'),('dimm-0051','temp1_crit_alarm'),
                         ('dimm-0051','temp1_lcrit_alarm'),('dimm-0053','temp1_fault'),
                         ('dimm-0053','unknown_alarm')]:
            r=reading(0);r['sensors'][key]['alarms']={name:1,'temp1_max_alarm':1}
            self.assertEqual(ht.ThermalGuard().observe(r,0),'host_thermal_hardware_alarm:'+key)

    def test_warning_admission_and_sticky_alarm_are_allowed(self):
        for temp in (55,59.75):
            g=ht.ThermalGuard()
            for t in range(2):
                r=reading(t,40,(temp,40));r['sensors']['dimm-0051']['alarms']={'temp1_max_alarm':1}
                self.assertIsNone(g.observe(r,t,admission=True))
            self.assertEqual(g.admission_safe_samples,2)
            self.assertIsNone(g.admission_cooldown_started)

    def test_admission_cools_longer_than_runtime_dwell_then_requires_two_safe_samples(self):
        g=ht.ThermalGuard()
        for t in range(40):
            self.assertIsNone(g.observe(reading(t,40,(60,61)),t,admission=True))
            self.assertEqual(g.admission_safe_samples,0)
            self.assertFalse(g.high_since)
        self.assertIsNone(g.observe(reading(40,40,(59,60)),40,admission=True))
        self.assertEqual(g.admission_safe_samples,0)
        self.assertIsNone(g.observe(reading(41,40,(59,59)),41,admission=True))
        self.assertEqual(g.admission_safe_samples,1)
        self.assertIsNone(g.observe(reading(42,40,(59,59)),42,admission=True))
        self.assertEqual(g.admission_safe_samples,2)
        # Admission heat cannot contaminate the post-release runtime timer.
        for t in range(43,68):self.assertIsNone(g.observe(reading(t,40,(60,40)),t))
        self.assertEqual(g.observe(reading(68,40,(60,40)),68),'host_thermal_sustained:dimm-0051')

    def test_admission_bound_cannot_be_reset_by_intermittent_cooling(self):
        g=ht.ThermalGuard()
        for t in range(120):
            temp=59 if t%3==1 else 60
            self.assertIsNone(g.observe(reading(t,40,(temp,40)),t,admission=True))
        self.assertEqual(g.observe(reading(120,40,(59,40)),120,admission=True),
                         'host_thermal_admission_cooldown_timeout')
        self.assertEqual(g.admission_cooldown_started,0)

    def test_admission_critical_missing_stale_fail_immediately_while_cooling(self):
        for problem in ('critical','alarm','missing','stale'):
            g=ht.ThermalGuard();g.observe(reading(0,40,(60,40)),0,admission=True)
            r=reading(1,40,(60,40))
            if problem=='critical':r['sensors']['dimm-0053']['temperature_c']=65
            if problem=='alarm':r['sensors']['dimm-0053']['alarms']={'temp1_crit_alarm':1}
            if problem=='missing':r['sensors'].pop('dimm-0053')
            if problem=='stale':r['started_monotonic']=-5
            if problem in ('missing','stale'):
                with self.assertRaises(ValueError):g.observe(r,1,admission=True)
            else:self.assertIn('dimm-0053',g.observe(r,1,admission=True))

    def test_admission_deadline_uses_wall_clock_not_sample_start(self):
        g=ht.ThermalGuard()
        for t in range(120):
            self.assertIsNone(g.observe(reading(t,40,(60,40)),t,admission=True))
        self.assertTrue(g.admission_expired(120))
        self.assertEqual(g.observe(reading(119.9,40,(59,40)),120.1,admission=True),
                         'host_thermal_admission_cooldown_timeout')

    def test_runtime_dimm_hysteresis_resets_at_58_and_warn_clears_at_53(self):
        g=ht.ThermalGuard();g.observe(reading(0,40,(60,40)),0)
        g.observe(reading(1,40,(59,40)),1);self.assertIn('dimm-0051',g.high_since)
        g.observe(reading(2,40,(58,40)),2);self.assertFalse(g.high_since)
        g.observe(reading(3,40,(54,40)),3);self.assertIn('dimm-0051',g.warnings)
        g.observe(reading(4,40,(53,40)),4);self.assertNotIn('dimm-0051',g.warnings)


class SysfsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.hw=self.root/'class';self.hw.mkdir()
        self.cpu=self.root/'devices/cpu';self.dimm=self.root/'devices/i2c-1'
        self.info=self.root/'cpuinfo';self.info.write_text('AMD Ryzen 9 9950X 16-Core Processor')
        self.paths={}
        for n,(key,dev) in enumerate([('cpu',self.cpu),('dimm-0051',self.dimm/'1-0051'),('dimm-0053',self.dimm/'1-0053')]):
            h=dev/'hwmon'/f'hwmon{n}';h.mkdir(parents=True);(self.hw/f'hwmon{n}').symlink_to(h)
            self.paths[key]=h
            values={'name':'k10temp' if key=='cpu' else 'spd5118','temp1_input':'40000'}
            if key=='cpu':values['temp1_label']='Tctl'
            else:
                b=bytearray(1024);b[521:551]=ht.MODULE_PART.ljust(30).encode();(dev/'eeprom').write_bytes(b)
                values.update(temp1_enable='1',temp1_max='55000',temp1_crit='85000',temp1_min='0',temp1_lcrit='0',
                              temp1_max_alarm='0',temp1_crit_alarm='0',temp1_min_alarm='0',temp1_lcrit_alarm='0')
            for k,v in values.items():(h/k).write_text(v)
        self.patchcpu=patch.object(ht,'CPU_DEVICE',str(self.cpu));self.patchcpu.start();self.addCleanup(self.patchcpu.stop)
        self.patchdimm=patch.object(ht,'DIMM_PARENT',str(self.dimm));self.patchdimm.start();self.addCleanup(self.patchdimm.stop)

    def sample(self):return ht.sample(self.hw,self.info)

    def test_identity_units_freshness_and_hwmon_renumbering(self):
        a=self.sample();b=self.sample()
        self.assertGreater(b['started_monotonic'],a['ended_monotonic'])
        self.assertEqual(set(a['sensors']),set(ht.SENSOR_IDS))
        self.assertEqual(a['sensors']['cpu']['temperature_c'],40)
        (self.hw/'hwmon1').rename(self.hw/'hwmon91')
        self.assertEqual(self.sample()['sensors'],a['sensors'])

    def test_read_failure_preserves_exact_raw_fields(self):
        (self.paths['dimm-0053']/'temp1_input').write_text('bad-value')
        with self.assertRaises(ht.SensorError) as caught:self.sample()
        self.assertIn('bad-value',json.dumps(caught.exception.evidence))

    def test_missing_disabled_changed_module_alarm_settings_and_cpu_fail(self):
        targets=[(self.paths['dimm-0051']/'temp1_enable','0'),
                 (self.paths['dimm-0051']/'temp1_max','65000'),
                 (self.paths['cpu']/'temp1_label','CPU socket'),
                 (self.info,'different processor')]
        for p,v in targets:
            old=p.read_text();p.write_text(v)
            with self.assertRaises(ht.SensorError):self.sample()
            p.write_text(old)
        (self.dimm/'1-0051/eeprom').write_bytes(bytes(1024))
        with self.assertRaisesRegex(ht.SensorError,'identity'):self.sample()
        (self.hw/'hwmon1').unlink()
        with self.assertRaisesRegex(ht.SensorError,'unavailable'):self.sample()

    def test_thermal_atomic_file_accounting_still_enforces_real_sizes(self):
        from test_evidence_sizes import rename_during_stat
        (self.root/'thermal-latest.json').write_text('x'*1000)
        self.assertEqual(evidence_sizes(self.root)['thermal-latest.json'],1000)
        self.assertGreaterEqual(sum(rename_during_stat(self.root).values()),1000)
