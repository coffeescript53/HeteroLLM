"""Selected-GPU sensors; normal AMD runtime resume, never power-setting writes.

Call only inside the pilot's two-second Reader. A driver open/read can block;
the independent supervisor deadline, not a Python timeout, bounds that case.
"""
import math
import os
from pathlib import Path
import stat
import time

from .hardware import parse_nvidia, query
from .supervisor import append_json, stamp

REQUIRED = ('total_bytes', 'used_bytes', 'temperature_c')
TRANSITION = ('suspended', 'suspending', 'resuming')


class SensorUnavailable(ValueError):
    def __init__(self, name, evidence, reason):
        self.evidence = evidence
        super().__init__(f'{name}: {reason}; ' + repr(evidence))


def read_raw(path):
    try:
        return {'path': str(path), 'value': path.read_text().strip(), 'error': None}
    except OSError as e:
        return {'path': str(path), 'value': None, 'error': str(e), 'errno': e.errno}


def number(record):
    try:
        return int(record['value'])
    except (TypeError, ValueError):
        return None


def problems(item):
    missing = [k for k in REQUIRED if item.get(k) is None]
    invalid = [k for k in REQUIRED if item.get(k) is not None and
               (not math.isfinite(item[k]) or item[k] < 0)]
    if not missing and not invalid and (item['total_bytes'] == 0 or item['used_bytes'] > item['total_bytes']):
        invalid.append('VRAM range')
    return {'missing_fields': missing, 'invalid_fields': invalid}


def amd_read(pci, pci_root=Path('/sys/bus/pci/devices')):
    dev = pci_root / pci
    raw = {k: read_raw(dev / k) for k in ('vendor', 'device', 'power/runtime_status',
           'power/control', 'power/autosuspend_delay_ms', 'mem_info_vram_total', 'mem_info_vram_used')}
    state = raw['power/runtime_status']['value']
    temps = {str(p): read_raw(p) for p in sorted(dev.glob('hwmon/hwmon*/temp*_input'))} if state == 'active' else {}
    raw['temperatures'] = temps
    raw['temperature_skip_reason'] = None if state == 'active' else 'runtime device not active'
    raw['gpu_busy_percent'] = read_raw(dev/'gpu_busy_percent') if state == 'active' else None
    raw['runtime_status_after'] = read_raw(dev / 'power/runtime_status')
    values = [number(r) for r in temps.values()]
    # Every exposed temperature channel must be readable; no partial or cached substitute.
    temperature = max(values) / 1000 if values and all(v is not None and v >= 0 for v in values) else None
    item = {'pci': pci, 'runtime_status': state, 'runtime_status_after': raw['runtime_status_after']['value'],
            'total_bytes': number(raw['mem_info_vram_total']), 'used_bytes': number(raw['mem_info_vram_used']),
            'temperature_c': temperature, 'busy_percent': number(raw['gpu_busy_percent']) if state == 'active' else None,
            'raw': raw, **stamp()}
    return item


def resume_render_node(pci, record, pci_root=Path('/sys/bus/pci/devices'), dev_root=Path('/dev/dri')):
    """One verified render-node open/close; no writes, ioctls or user GPU work."""
    dev = pci_root / pci
    if (read_raw(dev/'vendor')['value'], read_raw(dev/'device')['value']) != ('0x1002', '0x7551'):
        raise ValueError('selected PCI device is not the reviewed R9700')
    if (dev/'driver').resolve().name != 'amdgpu':
        raise ValueError('selected PCI driver is not amdgpu')
    nodes = sorted((dev/'drm').glob('renderD[0-9]*'))
    if len(nodes) != 1:
        raise ValueError('selected PCI render node missing or ambiguous')
    if (nodes[0]/'device').resolve() != dev.resolve():
        raise ValueError('render-node PCI identity mismatch')
    major, minor = map(int, (nodes[0]/'dev').read_text().strip().split(':'))
    path = dev_root / nodes[0].name
    expected = os.makedev(major, minor)
    s = path.stat()
    if not stat.S_ISCHR(s.st_mode) or s.st_rdev != expected:
        raise ValueError('render-node character-device identity mismatch')
    record('resume_open_intent', {'pci': pci, 'path': str(path), 'major': major, 'minor': minor})
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        if not stat.S_ISCHR(s.st_mode) or s.st_rdev != expected:
            raise ValueError('opened render-node identity mismatch')
    finally:
        os.close(fd)
    record('resume_open_complete', {'pci': pci, 'path': str(path)})


def amd_ready(pci, record, pci_root=Path('/sys/bus/pci/devices'), dev_root=Path('/dev/dri')):
    before = amd_read(pci, pci_root)
    record('amd_before', before)
    history = {'pci': pci, 'before': before, 'refresh_attempts': 0}
    issue = problems(before)
    if (before['raw']['vendor']['value'], before['raw']['device']['value']) != ('0x1002', '0x7551'):
        raise SensorUnavailable('ROCm0', {**history, **issue}, 'selected PCI device is not the reviewed R9700')
    if issue['invalid_fields'] or (before['temperature_c'] is not None and before['temperature_c'] >= 85):
        raise SensorUnavailable('ROCm0', {**history, **issue}, 'unsafe sensor values')
    if not any(issue.values()) and before['runtime_status_after'] == 'active':
        return before
    states = (before['runtime_status'], before['runtime_status_after'])
    if not any(s in TRANSITION for s in states):
        raise SensorUnavailable('ROCm0', {**history, **issue}, 'required sensor unavailable outside a runtime transition')
    history['refresh_attempts'] = 1
    try:
        resume_render_node(pci, record, pci_root, dev_root)
    except (OSError, ValueError) as e:
        raise SensorUnavailable('ROCm0', {**history, **issue, 'refresh_error': str(e),
                                        'errno': getattr(e, 'errno', None)}, 'runtime readiness refresh failed') from e
    deadline = time.monotonic() + .5
    while True:
        after = amd_read(pci, pci_root)
        record('amd_after', after)
        issue = problems(after)
        if not any(issue.values()) and after['runtime_status_after'] == 'active':
            return {**after, 'readiness_refresh': history}
        if after['runtime_status_after'] not in TRANSITION or time.monotonic() >= deadline:
            raise SensorUnavailable('ROCm0', {**history, 'after': after, **issue}, 'fresh required sensors unavailable after one refresh')
        time.sleep(.05)


def gpu_sample(devices, evidence_path=None):
    def record(event, data):
        if evidence_path is not None:
            append_json(evidence_path, {'event': event, **stamp(), 'data': data})
    record('sample_started', {'devices': devices})
    # Include optional utilization in the same bounded query as mandatory fields.
    raw = query(['nvidia-smi', '--query-gpu=pci.bus_id,name,memory.free,memory.total,memory.used,temperature.gpu,driver_version,utilization.gpu',
                 '--format=csv,noheader,nounits'], timeout=1)
    record('nvidia_raw', raw)
    rows = []
    for line in raw['stdout'].splitlines():
        fields = line.rsplit(',', 1)
        parsed = parse_nvidia(fields[0]) if len(fields) == 2 else []
        for item in parsed:
            try:
                item['utilization_percent'] = float(fields[1])
            except ValueError:
                item['utilization_percent'] = None
            rows.append(item)
    matches = [r for r in rows if r['pci'] == devices['CUDA0']]
    item = matches[0] if len(matches) == 1 else {}
    issue = problems(item)
    if raw['exit_code'] != 0 or len(matches) != 1 or any(issue.values()):
        raise SensorUnavailable('CUDA0', {'pci': devices['CUDA0'], **issue, 'raw': raw,
                                         'matching_devices': len(matches)}, 'required GPU sensors unavailable')
    return {'CUDA0': item, 'ROCm0': amd_ready(devices['ROCm0'], record)}
