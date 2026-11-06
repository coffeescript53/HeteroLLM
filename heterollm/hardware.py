"""Read-only runtime and Linux discovery; missing telemetry stays null."""
from dataclasses import asdict, dataclass
import csv
import hashlib
import io
import os
from pathlib import Path
import platform
import re
import subprocess
import time

MIB = 1024**2


def clean_environment(overrides=None, diagnostic=False):
    # Exclude hidden runtime overrides, credentials, and profiling flags.
    env = {k: os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','TMPDIR') if k in os.environ}
    env.update(overrides or {})
    if diagnostic:
        env['GGML_SCHED_DEBUG'] = '2'
    return env


def query(argv, env=None, timeout=20):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return {'argv':argv, 'exit_code':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'argv':argv, 'exit_code':None, 'stdout':'', 'stderr':str(e)}


def parse_meminfo(text):
    values = {}
    for line in text.splitlines():
        m = re.fullmatch(r'(\w+):\s+(\d+)\s+kB', line)
        if m:
            values[m[1]] = int(m[2]) * 1024
    if not {'MemTotal','MemAvailable','SwapTotal','SwapFree'} <= values.keys():
        raise ValueError('incomplete Linux memory counters')
    return values


@dataclass(frozen=True)
class Device:
    name: str
    description: str
    total_bytes: int
    free_bytes: int
    pci: str | None = None


def parse_devices(text):
    found = []
    for name, description, total, free in re.findall(r'^\s*((?:CUDA|ROCm)\d+): (.+) \((\d+) MiB, (\d+) MiB free\)', text, re.M):
        if int(free) > int(total):
            raise ValueError('runtime free memory exceeds total')
        found.append(Device(name, description, int(total)*MIB, int(free)*MIB))
    if len({x.name for x in found}) != len(found):
        raise ValueError('duplicate runtime device')
    return found


def parse_nvidia(text):
    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if len(fields) != 7:
            continue
        try:
            pci, name, free, total, used, temp, driver = [x.strip() for x in fields]
            rows.append({'pci':pci[-12:].lower(), 'name':name, 'free_bytes':int(free)*MIB,
                         'total_bytes':int(total)*MIB, 'used_bytes':int(used)*MIB,
                         'temperature_c':float(temp) if temp not in ('N/A','[N/A]') else None, 'driver':driver})
        except ValueError:
            continue
    return rows


def text_or_none(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def telemetry():
    result = {'time_ns':time.time_ns(), 'memory':parse_meminfo(Path('/proc/meminfo').read_text()), 'amd':[]}
    n = query(['nvidia-smi','--query-gpu=pci.bus_id,name,memory.free,memory.total,memory.used,temperature.gpu,driver_version','--format=csv,noheader,nounits'], timeout=5)
    result['nvidia'] = parse_nvidia(n['stdout']) if n['exit_code'] == 0 else []
    result['nvidia_error'] = n['stderr'] or n['stdout'] if n['exit_code'] != 0 else None
    for card in sorted(Path('/sys/class/drm').glob('card[0-9]*')):
        dev = card / 'device'
        if text_or_none(dev / 'vendor') != '0x1002':
            continue
        state = text_or_none(dev / 'power/runtime_status')
        item = {'pci':dev.resolve().name, 'runtime_status':state, 'total_bytes':None,
                'used_bytes':None, 'temperature_c':None, 'busy_percent':None}
        # Do not touch sleeping-device sensors or any ECC counters.
        for key, filename in (('total_bytes','mem_info_vram_total'), ('used_bytes','mem_info_vram_used')):
            value = text_or_none(dev / filename)
            item[key] = int(value) if value and value.isdigit() else None
        if state == 'active':
            temps = [text_or_none(p) for p in dev.glob('hwmon/hwmon*/temp*_input')]
            values = [int(v)/1000 for v in temps if v and v.lstrip('-').isdigit()]
            item['temperature_c'] = max(values) if values else None
            busy = text_or_none(dev / 'gpu_busy_percent')
            item['busy_percent'] = int(busy) if busy and busy.isdigit() else None
        result['amd'].append(item)
    return result


def discover(config):
    env = clean_environment(config.environment)
    listing = query([config.cli, '--list-devices'], env)
    bench = query([config.bench, '--help'], env)
    cli = query([config.cli, '--help'], env)
    versions = {name:query([path,'--version'],env) for name,path in (('bench',config.bench),('cli',config.cli))}
    source = query(['git','-C',config.source,'rev-parse','HEAD'])
    status = query(['git','-C',config.source,'status','--porcelain'])
    snap = telemetry()
    devices = []
    for d in parse_devices(listing['stdout']):
        pci = None
        if d.name.startswith('CUDA'):
            matches = [n for n in snap['nvidia'] if n['name'] == d.description]
            if len(matches) == 1:
                pci = matches[0]['pci']
        elif 'R9700' in d.description:
            matches = [a for a in snap['amd'] if text_or_none(Path('/sys/bus/pci/devices') / a['pci'] / 'device') == '0x7551']
            if len(matches) == 1:
                pci = matches[0]['pci']
        devices.append(asdict(Device(d.name,d.description,d.total_bytes,d.free_bytes,pci)))
    return {'timestamp_ns':time.time_ns(), 'platform':platform.platform(), 'cpu':query(['lscpu']),
            'ram_channel_mode':'unknown', 'devices':devices, 'telemetry':snap, 'listing':listing,
            'bench_help':bench, 'cli_help':cli, 'versions':versions, 'source':source, 'source_status':status}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def file_identity(path, hash_bytes=True):
    p = Path(path).resolve()
    s = p.stat()
    return {'path':str(p), 'size_bytes':s.st_size, 'mtime_ns':s.st_mtime_ns,
            'sha256':sha256_file(p) if hash_bytes else None}


def runtime_identity(config):
    paths = {Path(config.bench).resolve(), Path(config.cli).resolve()}
    for directory in {Path(config.bench).parent,Path(config.cli).parent}:
        paths.update(p.resolve() for p in directory.glob('*.so*') if p.is_file())
    return [file_identity(p) for p in sorted(paths)]
