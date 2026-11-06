"""Read-only, pinned Ryzen 9950X and two DDR5 SPD-hub thermal protection.

No motherboard/SSD fallback. SPD-hub readings are not DRAM die temperatures.
DIMM 55/60/65 C thresholds are provisional engineering safeguards, not
manufacturer-certified limits. See docs/DDR5_THERMAL_CORRECTION.md.
"""
import math
from pathlib import Path
import time

CPU_DEVICE = '/sys/devices/pci0000:00/0000:00:18.3'
DIMM_PARENT = '/sys/devices/pci0000:00/0000:00:14.0/i2c-1'
MODULE_PART = 'F5-6000J2836F48G'
SENSOR_IDS = ('cpu', 'dimm-0051', 'dimm-0053')
POLICY = {
    'cpu': {'warning_c': 90, 'stop_c': 95, 'critical_c': 95,
            'critical_inclusive': False, 'hysteresis_c': 2},
    'dimm': {'warning_c': 55, 'stop_c': 60, 'critical_c': 65,
             'critical_inclusive': True, 'hysteresis_c': 2},
}
SUSTAINED_SECONDS = 25
MAX_AGE_SECONDS = 3
DIMM_ADMISSION_COOLDOWN_SECONDS = 120


class SensorError(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def sample(root=Path('/sys/class/hwmon'), cpuinfo=Path('/proc/cpuinfo')):
    """Fresh read every call, no cached values; all sysfs handles opened read-only."""
    evidence = {'started_monotonic': time.monotonic(), 'sensors': {}, 'raw': {}}
    try:
        if 'AMD Ryzen 9 9950X 16-Core Processor' not in cpuinfo.read_text():
            raise ValueError('unverified CPU model')
        for hw in sorted(Path(root).glob('hwmon*')):
            driver = (hw/'name').read_text().strip()
            if driver not in ('k10temp', 'spd5118'):
                continue
            device = hw.resolve().parent.parent
            raw = {'hwmon': str(hw), 'device': str(device), 'driver': driver}
            evidence['raw'][str(hw)] = raw
            # Identity is the physical device path, not reboot-dependent hwmonN.
            if driver == 'k10temp':
                if str(device) != CPU_DEVICE or (hw/'temp1_label').read_text().strip() != 'Tctl':
                    raise ValueError('unverified CPU sensor identity')
                key, kind = 'cpu', 'cpu'
            else:
                if str(device) not in (DIMM_PARENT+'/1-0051', DIMM_PARENT+'/1-0053'):
                    raise ValueError('unverified or extra physical DIMM sensor')
                key, kind = 'dimm-'+device.name[-4:], 'dimm'
                with (device/'eeprom').open('rb') as f:
                    f.seek(521)
                    raw['module_part'] = f.read(30).decode('ascii').strip(' \x00')
                if raw['module_part'] != MODULE_PART:
                    raise ValueError('DIMM module identity changed')
            if key in evidence['sensors']:
                raise ValueError('duplicate thermal sensor')
            fields = ['temp1_input']
            if kind == 'dimm':
                fields += ['temp1_enable', 'temp1_max', 'temp1_crit', 'temp1_min', 'temp1_lcrit',
                           'temp1_max_alarm', 'temp1_crit_alarm', 'temp1_min_alarm', 'temp1_lcrit_alarm']
            else:
                fields += [p.name for p in hw.glob('temp1_*')
                           if p.name.endswith(('_alarm', '_fault')) or p.name == 'temp1_alarm']
            for name in sorted(set(fields)):
                raw[name] = (hw/name).read_text().strip()
            values = {k: int(raw[k]) for k in fields}
            if kind == 'dimm' and (values['temp1_enable'] != 1 or
                    values['temp1_max'] != 55000 or values['temp1_crit'] != 85000 or
                    values['temp1_min'] != 0 or values['temp1_lcrit'] != 0):
                raise ValueError('DIMM sensor enable/alarm settings changed')
            alarms = {k:v for k,v in values.items() if k.endswith(('_alarm', '_fault'))}
            if any(v not in (0, 1) for v in alarms.values()):
                raise ValueError('invalid hardware alarm')
            evidence['sensors'][key] = {'kind':kind, 'identity':str(device),
                'driver':driver, 'temperature_c':values['temp1_input']/1000,
                'alarms':alarms, 'unit':'C', 'raw_unit':'millidegree C'}
        if set(evidence['sensors']) != set(SENSOR_IDS):
            raise ValueError('required CPU or physical DIMM sensor unavailable')
        evidence['ended_monotonic'] = time.monotonic()
        return evidence
    except Exception as e:
        evidence['ended_monotonic'] = time.monotonic()
        raise SensorError(str(e), evidence) from e


class ThermalGuard:
    """Per-sensor persistence; repeated polling of one record never advances it."""
    def __init__(self):
        self.previous = None
        self.identities = None
        self.high_since = {}
        self.warnings = set()
        self.maxima = {}
        self.count = 0
        self.transitions = []
        self.high_alarm_warnings = {}
        self.admission_cooldown_started = None
        self.admission_safe_samples = 0

    def observe(self, data, now, *, admission=False):
        self.transitions = []
        start, end = data['started_monotonic'], data['ended_monotonic']
        if (not all(type(t) in (int, float) and math.isfinite(t) for t in (start,end,now))
                or not start <= end <= now or now-start > MAX_AGE_SECONDS
                or (self.previous is not None and (start <= self.previous or start-self.previous > MAX_AGE_SECONDS))):
            raise ValueError('host thermal stale/invalid sample timestamp')
        sensors = data['sensors']
        if set(sensors) != set(SENSOR_IDS):
            raise ValueError('required host thermal sensor missing')
        identities = {k:[v['identity'],v['driver'],v['kind']] for k,v in sensors.items()}
        if self.identities is not None and identities != self.identities:
            raise ValueError('host thermal sensor identity changed')
        # Validate the entire record before using it as safety evidence.
        for key, r in sensors.items():
            expected_kind = 'cpu' if key == 'cpu' else 'dimm'
            v = r['temperature_c']
            if (r['kind'] != expected_kind or r['unit'] != 'C' or
                    type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 125 or
                    not isinstance(r['alarms'],dict) or
                    any(type(a) is not int or a not in (0,1) for a in r['alarms'].values())):
                raise ValueError('invalid host thermal reading: '+key)
        self.identities = identities
        self.previous = start
        self.count += 1
        reason = None
        for key, r in sensors.items():
            v = r['temperature_c']; p = POLICY[r['kind']]
            self.maxima[key] = max(self.maxima.get(key, v), v)
            high_alarm = r['kind'] == 'dimm' and bool(r['alarms'].get('temp1_max_alarm', 0))
            if r['kind'] == 'dimm':
                previous = self.high_alarm_warnings.get(key, {})
                self.high_alarm_warnings[key] = {
                    'asserted': high_alarm,
                    'asserted_samples': previous.get('asserted_samples', 0) + int(high_alarm),
                    'may_be_sticky': True,
                    'current_temperature_below_warning': v < p['warning_c'],
                }
            # Only the DIMM high-warning bit is noncritical. CPU alarms, critical
            # high/low alarms, faults and unknown asserted conditions fail closed.
            if any(a for name,a in r['alarms'].items()
                   if not (r['kind'] == 'dimm' and name == 'temp1_max_alarm')):
                reason = reason or 'host_thermal_hardware_alarm:'+key
            if v > p['critical_c'] or (p['critical_inclusive'] and v == p['critical_c']):
                reason = reason or 'host_thermal_critical:'+key
            if admission and r['kind'] == 'cpu' and v >= p['warning_c']:
                reason = reason or 'host_thermal_admission:'+key
            if (v >= p['warning_c'] or high_alarm) and key not in self.warnings:
                self.warnings.add(key)
                self.transitions.append({'sensor':key,'state':'warning','temperature_c':v,
                                         'high_alarm':high_alarm,'high_alarm_may_be_sticky':high_alarm})
            elif not high_alarm and v <= p['warning_c']-p['hysteresis_c'] and key in self.warnings:
                self.warnings.remove(key)
                self.transitions.append({'sensor':key,'state':'recovered','temperature_c':v})
            if admission and r['kind'] == 'dimm':
                # Cooling a gated process is not runtime overheating. Its separate
                # absolute bound below never resets on brief temperature dips.
                self.high_since.pop(key, None)
                continue
            if v >= p['stop_c']:
                self.high_since.setdefault(key, start)
            elif v <= p['stop_c']-p['hysteresis_c']:
                self.high_since.pop(key, None)
            if key in self.high_since and start-self.high_since[key] >= SUSTAINED_SECONDS:
                reason = reason or 'host_thermal_sustained:'+key
        if admission:
            hot = [k for k,r in sensors.items()
                   if r['kind'] == 'dimm' and r['temperature_c'] >= POLICY['dimm']['stop_c']]
            if hot:
                self.admission_safe_samples = 0
                if self.admission_cooldown_started is None:
                    self.admission_cooldown_started = start
                    self.transitions.append({'state':'dimm_admission_cooling','sensors':hot,
                                             'maximum_seconds':DIMM_ADMISSION_COOLDOWN_SECONDS})
            else:
                self.admission_safe_samples += 1
            if self.admission_expired(now):
                reason = reason or 'host_thermal_admission_cooldown_timeout'
        return reason

    def admission_expired(self, now):
        return (self.admission_cooldown_started is not None and
                now-self.admission_cooldown_started >= DIMM_ADMISSION_COOLDOWN_SECONDS)

    def summary(self):
        return {'valid_samples':self.count, 'max_observed_c':self.maxima,
                'sensor_identity':self.identities, 'warnings':sorted(self.warnings),
                'high_since_monotonic':self.high_since, 'policy':POLICY,
                'high_alarm_warnings':self.high_alarm_warnings,
                'admission_safe_samples':self.admission_safe_samples,
                'admission_cooldown_started_monotonic':self.admission_cooldown_started,
                'admission_cooldown_max_seconds':DIMM_ADMISSION_COOLDOWN_SECONDS,
                'sustained_seconds':SUSTAINED_SECONDS, 'maximum_age_seconds':MAX_AGE_SECONDS,
                'dimm_policy_basis':'provisional engineering safeguards; high alarm is warning-only; not manufacturer-certified operating limits'}
