"""One reviewed 96 GB functional pilot. Preview by default; no benchmark path."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import re
import shutil
import time

from .config import Config, Policy
from .gguf import read_layout
from .gpu_sensors import gpu_sample
from .hardware import (clean_environment, discover, file_identity, parse_meminfo,
                       query, runtime_identity)
from .planner import estimate
from .runner import execution_lock, load_receipt
from .runtime import functional_command, placement, require_capabilities, response_correct
from .supervisor import Limits, Reader, supervise

GIB = 1024**3
POLICY = Policy('mixed-equal', ('CUDA0','ROCm0'), (1,1), 999)


def command(config):
    argv = functional_command(config, POLICY)
    argv[argv.index('-lm')+1] = 'none'
    return argv


def process_sample(pid):
    proc = Path('/proc')/str(pid)
    raw = (proc/'status').read_text()
    stat = (proc/'stat').read_text()
    fields = stat.rsplit(')',1)[1].split()
    values = {k:int(v)*1024 for k,v in re.findall(r'^(\w+):\s+(\d+) kB$',raw,re.M)}
    required = {'VmRSS','VmHWM','VmSwap','VmLck','VmPin','VmPTE','RssAnon','RssFile','RssShmem'}
    if fields[0] in ('Z','X'):
        for k in required:
            values.setdefault(k,0)
    if not required <= values.keys():
        raise ValueError('incomplete process memory counters')
    return {**{k:values[k] for k in required}, 'pid':pid, 'state':fields[0],
            'start_ticks':int(fields[19]), 'minor_faults':int(fields[7]),
            'major_faults':int(fields[9]), 'user_ticks':int(fields[11]),
            'system_ticks':int(fields[12]), 'status_raw':raw, 'stat_raw':stat}


def fast_sample(pid):
    mem = Path('/proc/meminfo').read_text()
    values = parse_meminfo(mem)
    vmstat = Path('/proc/vmstat').read_text()
    vmvalues = dict(line.split() for line in vmstat.splitlines())
    return {'memory':values, 'swap_used':values['SwapTotal']-values['SwapFree'],
            'meminfo_raw':mem, 'vmstat_raw':vmstat, 'process':process_sample(pid),
            'pressure_raw':Path('/proc/pressure/memory').read_text(),
            'oom_kill':int(vmvalues['oom_kill'])}


def maps_sample(pid, model):
    proc = Path('/proc')/str(pid)
    try:
        rollup = (proc/'smaps_rollup').read_text()
        # Preserve only the approved model's VMAs, not unrelated file paths.
        blocks = re.split(r'(?=^[0-9a-f]+-[0-9a-f]+ )', (proc/'smaps').read_text(), flags=re.M)
        found = [b for b in blocks if str(model) in b.split('\n',1)[0]]
        return {'smaps_rollup_raw':rollup, 'model_mappings':found,
                'process_start_ticks':process_sample(pid)['start_ticks']}
    except (FileNotFoundError, ProcessLookupError):
        if process_sample(pid)['state'] in ('Z','X'):
            return {'smaps_rollup_raw':None, 'model_mappings':[], 'state':'zombie'}
        raise


def bounded_gpu_readiness(config):
    reader = Reader('preflight-gpu', partial(gpu_sample, config.devices,
                    Path(config.output)/'preflight-gpu-raw.jsonl'), config.output, None, 2)
    try:
        while True:
            value = reader.poll()
            if value is not None:
                return value['data']
            time.sleep(.02)
    finally:
        if not reader.close():
            raise RuntimeError('preflight GPU reader cleanup incomplete')


def sampler_factory(config, pid):
    return {'fast':partial(fast_sample,pid), 'gpu':partial(gpu_sample,config.devices,
            Path(config.output)/'gpu-raw.jsonl'),
            'maps':partial(maps_sample,pid,config.model)}


def dependencies(config):
    records = {}; files = {}
    paths = [Path(config.cli), *sorted(Path(config.cli).parent.glob('libggml-*.so'))]
    for p in paths:
        answer = query(['ldd',str(p)],clean_environment(config.environment),timeout=5)
        records[str(p)] = answer
        if answer['exit_code'] != 0 or 'not found' in answer['stdout']:
            raise ValueError('unresolved shared library: '+str(p))
        for path in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(',answer['stdout'],re.M):
            resolved = str(Path(path).resolve())
            if resolved not in files:
                files[resolved] = file_identity(resolved)
    return {'resolution':records,'files':files}


def preflight(config, plan):
    receipt = load_receipt(config,plan['receipt'])
    current = runtime_identity(config)
    saved = json.loads(Path(plan['runtime_baseline']).read_text())
    if current != saved:
        raise ValueError('runtime files changed from original Stage B')
    deps = dependencies(config)
    if deps['files'] != json.loads(Path(plan['dependency_baseline']).read_text())['files']:
        raise ValueError('resolved dependency files changed from 96 GB preparation')
    snap = discover(config)  # only --help / --version / --list-devices, never -m
    require_capabilities(snap)
    if snap['source']['exit_code'] != 0 or snap['source']['stdout'].strip() != config.source_revision:
        raise ValueError('source revision mismatch')
    if snap['source_status']['exit_code'] != 0 or snap['source_status']['stdout'].strip():
        raise ValueError('pinned source is dirty')
    for name, version in snap['versions'].items():
        commit = re.search(r'commit ([0-9a-f]{7,40})',version['stdout']+version['stderr'])
        if version['exit_code'] != 0 or not commit or not config.source_revision.startswith(commit[1]):
            raise ValueError(name+': binary version mismatch')
    memory = snap['telemetry']['memory']
    if memory['MemTotal'] != plan['mem_total_bytes'] or memory['MemAvailable'] < 16*GIB:
        raise ValueError('RAM configuration changed or admission RAM <16 GiB')
    if shutil.disk_usage(Path(config.output).parent).free < 2*GIB:
        raise ValueError('output disk free <2 GiB')
    layout = read_layout(config.model)
    screen = estimate(POLICY,layout,snap,config)
    if not screen.admissible:
        raise ValueError('GPU capacity/identity screening: '+str(screen.reasons))
    fast_sample(os.getpid()); maps_sample(os.getpid(),config.model)
    gpu = bounded_gpu_readiness(config)
    if any(g['temperature_c']>=85 for g in gpu.values()):
        raise ValueError('preflight temperature limit')
    idle = []; started = time.monotonic()
    while time.monotonic()-started < 10:
        m = parse_meminfo(Path('/proc/meminfo').read_text())
        used = m['SwapTotal']-m['SwapFree']
        if idle and used > idle[-1]['swap_used']:
            raise ValueError('swap increasing during idle admission window')
        if m['MemAvailable'] < 16*GIB or m.get('HardwareCorrupted',0):
            raise ValueError('idle RAM/hardware memory failure')
        idle.append({'monotonic':time.monotonic(),'swap_used':used,'available':m['MemAvailable']})
        time.sleep(.25)
    return {'model_receipt':receipt,'runtime_files':current,'dependencies':deps,'hardware':snap,
            'screening':screen.to_dict(),'idle_samples':idle,'swap_baseline':idle[-1]['swap_used'],
            'gpu_readiness':gpu,'hash_scope':'Model path/size/mtime rechecked against prior full SHA-256 receipt'}


def validate(stdout, stderr):
    evidence = placement(stderr)
    errors = []
    if not response_correct(stdout, stderr):
        errors.append('complete assistant response must be exactly 42; framed CLI output requires matching final message evidence')
    expected = {str(i):'CUDA0' if i<41 else 'ROCm0' for i in range(81)}
    if evidence['layers'] != expected or evidence['conflicting_layers']:
        errors.append('expected complete 0-40 CUDA0 / 41-80 ROCm0 map missing')
    for d in POLICY.devices:
        if evidence['model_buffers_mib'].get(d,0)<=0 or d not in evidence['matrix_compute_backends']:
            errors.append(d+': positive model allocation and matrix scheduling required')
    for d,pci in (('CUDA0','0000:05:00.0'),('ROCm0','0000:03:00.0')):
        identities = re.findall(r'using device '+d+r' \([^\n]*?\) \(([^)]+)\)',stderr)
        if not identities or set(identities)!={pci}:
            errors.append(d+': actual runtime physical PCI evidence missing or mismatched')
    if 512 not in evidence['contexts'] or not evidence['kv_lines']:
        errors.append('effective context 512 / KV allocation missing')
    if not re.search(r'load_mode\s*=\s*none\b',stderr):
        errors.append('load_mode = none not observed')
    # Includes CUDA_Host/ROCm_Host reports omitted by the generic placement parser.
    evidence['all_model_buffer_lines'] = [s for s in stderr.splitlines() if 'model buffer size' in s]
    evidence['loading_lines'] = [s for s in stderr.splitlines() if any(t in s.lower() for t in
                                ('async','staging','load_mode','lazy_mode','host buffer'))]
    return {'errors':errors,'placement':evidence}


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    config = Config.load(plan['config'])
    if (config.batch,config.ubatch,config.threads,config.prompt_tokens,config.generation_tokens,config.depth)!=(512,128,16,512,128,0):
        raise ValueError('unreviewed workload controls')
    if config.devices != {'CUDA0':'0000:05:00.0','ROCm0':'0000:03:00.0'}:
        raise ValueError('unreviewed physical GPUs')
    if config.model_sha256!='32df3baccb556f9840059b2528b2dee4d3d516b24afdfb9d0c56ff5f63e3a664' or config.model_size_bytes!=42520398816:
        raise ValueError('unreviewed model')
    if config.source_revision!='5e03bdd8700948b9c41c54dd1b00f28a2aebc03f':
        raise ValueError('unreviewed source')
    if config.environment != {'LD_LIBRARY_PATH':'/opt/rocm/core-10.0/lib'}:
        raise ValueError('unreviewed environment overrides')
    if (config.gpu_reserve_mib,config.host_reserve_mib)!=(3072,8192):
        raise ValueError('unreviewed planning reserves')
    return config,plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--execute',action='store_true',help='Requires explicit approval for this ONE attempt')
    args = parser.parse_args()
    config,plan = load_plan(args.plan)
    if not args.execute:
        print(json.dumps({'mode':'preview_only','argv':command(config),'output':config.output,
                          'limits':Limits().__dict__,'model_loading_processes':1},indent=2))
        return 0
    with execution_lock():
        result = supervise(command(config), config.output, clean_environment(config.environment,True),
                           partial(preflight,config,plan), partial(sampler_factory,config),validate)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
