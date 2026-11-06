"""Stage C0 only: four supervised diagnostics, preview by default. No C1 path."""
import argparse
from dataclasses import asdict, replace
from functools import partial
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import time

from .analysis import parse_benchmark
from .config import Policy
from .gguf import read_layout
from .hardware import discover, file_identity, parse_meminfo, runtime_identity
from .pilot import (bounded_gpu_readiness, dependencies, fast_sample, load_plan as load_pilot_plan,
                    maps_sample, sampler_factory)
from .planner import estimate
from .runner import execution_lock, load_receipt
from .runtime import bench_command, placement, require_capabilities, verify_placement
from .supervisor import Limits, admission_reason, atomic_json, append_json, stamp, supervise

GIB = 1024**3
MEM_TOTAL_TOLERANCE_BYTES = 1024**2
ROOT = Path(__file__).resolve().parents[1]
POLICIES = (
    Policy('mixed-equal', ('CUDA0', 'ROCm0'), (1, 1), 999),
    Policy('mixed-default', ('CUDA0', 'ROCm0'), None, 999),
    Policy('cuda-only', ('CUDA0',), None, 56),
    Policy('rocm-only', ('ROCm0',), None, 59),
)
C0_LIMITS = replace(Limits(), rss_cap=32*GIB, admission_ram=40*GIB, admission_temperature=45)
SUITE_SECONDS = 4200
COOLDOWN_SECONDS = 30
IDLE_SECONDS = 10


def command(config, policy):
    argv = bench_command(config, policy, 1, diagnostic=True)
    argv[argv.index('-lm')+1] = 'none'
    return argv + ['-fitc', '0', '-embd', '0', '-ncmoe', '0', '-mg', '0',
                   '--prio', '0', '--delay', '0', '--poll', '50', '--cpu-strict', '0', '-C', '0x0']


def environment(config):
    return {'HOME': str(Path.home()), 'LANG': 'en_US.UTF-8', 'LC_ALL': 'C.UTF-8',
            'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            **config.environment, 'GGML_SCHED_DEBUG': '2'}


def host_controls():
    """Read configured controls, not changing dynamic clocks or power policy."""
    paths = [p for name in ('scaling_governor', 'scaling_driver', 'scaling_min_freq', 'scaling_max_freq')
             for p in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/'+name))]
    paths += [Path('/sys/bus/pci/devices')/pci/'power/control'
              for pci in ('0000:05:00.0', '0000:03:00.0')]
    return {str(p): p.read_text().strip() for p in paths}


def hardware_signature(snapshot):
    prefixes = ('Architecture:', 'CPU(s):', 'On-line CPU(s) list:', 'Model name:',
                'Thread(s) per core:', 'Core(s) per socket:', 'Socket(s):', 'NUMA node(s):')
    return {'platform': snapshot['platform'],
            'cpu': [s for s in snapshot['cpu']['stdout'].splitlines() if s.startswith(prefixes)],
            'mem_total_bytes': snapshot['telemetry']['memory'].get('MemTotal'),
            'swap_total_bytes': snapshot['telemetry']['memory']['SwapTotal'],
            'devices': [{k: d[k] for k in ('name', 'description', 'pci')}
                        for d in snapshot['devices'] if d['name'] in ('CUDA0', 'ROCm0')],
            'nvidia_drivers': [{k: d[k] for k in ('pci', 'driver')}
                               for d in snapshot['telemetry']['nvidia']]}


def mem_total_comparison(original, observed):
    """Allow bounded reboot accounting drift, never a changed RAM baseline."""
    valid = type(original) is int and type(observed) is int and original > 0 and observed > 0
    difference = observed-original if valid else None
    accepted = valid and abs(difference) <= MEM_TOTAL_TOLERANCE_BYTES
    return {'original_bytes': original, 'observed_bytes': observed,
            'signed_difference_bytes': difference, 'tolerance_bytes': MEM_TOTAL_TOLERANCE_BYTES,
            'accepted': accepted, 'decision': 'accept' if accepted else 'reject'}


def concurrent_runtimes(proc=Path('/proc')):
    """Record only known llama process names/PIDs; never collect other argv."""
    found = []
    for path in proc.glob('[0-9]*/comm'):
        try:
            name = path.read_text().strip()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if name in ('llama-cli', 'llama-bench', 'llama-server', 'llama-run'):
            found.append({'pid': int(path.parent.name), 'name': name})
    return found


def load_plan(path):
    config, plan = load_pilot_plan(path)
    required = {'stage', 'config', 'receipt', 'runtime_baseline', 'dependency_baseline',
                'hardware_baseline', 'mem_total_bytes', 'authorization'}
    if set(plan) != required or plan['stage'] != 'C0' or plan['mem_total_bytes'] != 98774167552:
        raise ValueError('only the pinned four-diagnostic C0 plan is supported')
    expected_paths = {'source': ROOT/'llama.cpp', 'bench': ROOT/'llama.cpp/build-mixed/bin/llama-bench',
                      'cli': ROOT/'llama.cpp/build-mixed/bin/llama-cli',
                      'model': ROOT/'models/llama33-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf'}
    if any(Path(getattr(config, k)) != v for k, v in expected_paths.items()):
        raise ValueError('unreviewed model/runtime paths')
    if (config.max_launches, config.timeout_seconds, config.suite_seconds,
        config.max_swap_growth_mib, config.minimum_available_ram_mib,
        config.max_temperature_c, config.max_log_mib) != (4, 900, 4200, 256, 4096, 85, 256):
        raise ValueError('unreviewed C0 execution controls')
    output = Path(config.output)
    if output.parent != ROOT/'runs/llama33-70b' or not output.name.startswith('stage-c0-'):
        raise ValueError('C0 requires a dedicated fresh output path')
    return config, plan


def preview(config):
    return {'mode': 'preview_only', 'stage': 'C0', 'timings_eligible_for_comparison': False,
            'limits': asdict(C0_LIMITS), 'suite_seconds': SUITE_SECONDS,
            'output': config.output, 'output_exists': Path(config.output).exists(),
            'launches': [{'ordinal': i, 'policy': p.to_dict(), 'argv': command(config, p),
                          'environment': environment(config),
                          'output': str(Path(config.output)/f'{i:02d}-{p.name}')}
                         for i, p in enumerate(POLICIES, 1)]}


def runtime_preflight(config, plan, save):
    runtime = save('runtime-files.json', runtime_identity(config))
    if runtime != json.loads(Path(plan['runtime_baseline']).read_text()):
        raise ValueError('runtime files changed from verified Stage B')
    deps = save('dependencies.json', dependencies(config))
    if deps['files'] != json.loads(Path(plan['dependency_baseline']).read_text())['files']:
        raise ValueError('runtime dependencies changed')
    snapshot = save('hardware.json', discover(config))  # help/version/list only
    require_capabilities(snapshot)
    if (snapshot['source']['exit_code'] != 0 or snapshot['source']['stdout'].strip() != config.source_revision
            or snapshot['source_status']['exit_code'] != 0 or snapshot['source_status']['stdout'].strip()):
        raise ValueError('pinned source revision/status changed')
    for name, record in snapshot['versions'].items():
        commit = re.search(r'commit ([0-9a-f]{7,40})', record['stdout']+record['stderr'])
        if record['exit_code'] != 0 or not commit or not config.source_revision.startswith(commit[1]):
            raise ValueError(name+': binary version mismatch')
    return snapshot


def preflight(config, plan, policy, suite_swap_baseline, *, runtime_check=None):
    directory = Path(config.output)
    def save(name, value):
        atomic_json(directory/name, value)
        return value
    if save('concurrent-runtimes.json', concurrent_runtimes()):
        raise ValueError('another llama workload is active')
    receipt = save('model-receipt.json', load_receipt(config, plan['receipt']))
    snapshot = (runtime_check or runtime_preflight)(config, plan, save)
    controls = save('host-controls.json', host_controls())
    signature = save('hardware-signature.json', hardware_signature(snapshot))
    baseline = json.loads(Path(plan['hardware_baseline']).read_text())
    memory = snapshot['telemetry']['memory']
    memory_identity = save('memory-identity.json', {
        'hardware_baseline': mem_total_comparison(baseline['signature'].get('mem_total_bytes'),
                                                signature.get('mem_total_bytes')),
        'pinned_plan': mem_total_comparison(plan.get('mem_total_bytes'), memory.get('MemTotal'))})
    # Only MemTotal has a tolerance; preserve every other field and key exactly.
    if (not memory_identity['hardware_baseline']['accepted']
            or {k: v for k, v in signature.items() if k != 'mem_total_bytes'}
            != {k: v for k, v in baseline['signature'].items() if k != 'mem_total_bytes'}
            or controls != baseline['controls']):
        raise ValueError('96 GB hardware/driver/configured-control identity changed')
    for name, pci in config.devices.items():
        found = [d for d in snapshot['devices'] if d['name'] == name and d['pci'] == pci]
        if len(found) != 1:
            raise ValueError(name+': required physical GPU missing or mismatched')
    if not memory_identity['pinned_plan']['accepted'] or memory['MemAvailable'] < C0_LIMITS.admission_ram:
        raise ValueError('C0 admission needs the same RAM configuration and >=40 GiB available')
    if shutil.disk_usage(directory).free < 10*GIB:
        raise ValueError('C0 requires >=10 GiB free evidence disk space')
    layout = read_layout(config.model)
    save('gguf-layout.json', layout.to_dict())
    screen = estimate(policy, layout, snapshot, config)
    save('capacity-screen.json', screen.to_dict())
    if not screen.admissible:
        raise ValueError('fixed policy fails capacity screening: '+str(screen.reasons))
    # Pair the stricter runtime capacity estimate with the discovery sensor usage.
    sensor_used = {'CUDA0': next(n['used_bytes'] for n in snapshot['telemetry']['nvidia']
                                if n['pci'] == config.devices['CUDA0']),
                   'ROCm0': next(n['used_bytes'] for n in snapshot['telemetry']['amd']
                                 if n['pci'] == config.devices['ROCm0'])}
    capacity = {d: {'runtime_free_bytes': next(x['free_bytes'] for x in snapshot['devices'] if x['name'] == d),
                    'sensor_used_bytes': sensor_used[d]} for d in policy.devices}
    if any(type(v['sensor_used_bytes']) is not int for v in capacity.values()):
        raise ValueError('missing paired capacity sensor baseline')
    save('capacity-baseline.json', capacity)
    fast_sample(os.getpid()); maps_sample(os.getpid(), config.model)
    gpu = bounded_gpu_readiness(config)
    result = {'model_receipt': receipt, 'screening': screen.to_dict(), 'capacity_baseline': capacity,
              'policy': policy.to_dict(), 'swap_baseline': suite_swap_baseline,
              'gpu_readiness': gpu, 'idle_samples': [],
              'model_integrity_scope': 'Path/size/mtime against prior full SHA-256 receipt'}
    save('gpu-readiness.json', gpu)
    admission = admission_reason({'memory': memory}, gpu, result, C0_LIMITS)
    if admission:
        raise ValueError('fresh C0 admission failed: '+admission)
    started = time.monotonic()
    while time.monotonic()-started < IDLE_SECONDS:
        current = parse_meminfo(Path('/proc/meminfo').read_text())
        used = current['SwapTotal']-current['SwapFree']
        idle = result['idle_samples']
        if used-suite_swap_baseline > C0_LIMITS.swap_growth or (idle and used > idle[-1]['swap_used']):
            raise ValueError('swap increasing or suite swap limit exceeded during admission')
        if current['MemAvailable'] < C0_LIMITS.admission_ram or current.get('HardwareCorrupted', 0):
            raise ValueError('idle RAM/hardware memory failure')
        item = {'swap_used': used, 'available': current['MemAvailable'], **stamp()}
        idle.append(item); append_json(directory/'idle.jsonl', item)
        time.sleep(.25)
    return result


def strict_json(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON field: '+key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def validate_rows(stdout, config, policy, *, reps=1):
    rows = strict_json(stdout)
    # Reuse duration/workload checks while explicitly selecting the none contract.
    parse_benchmark(rows, config, reps, load_mode='none')
    for row in rows:
        expected = {'model_filename': config.model, 'devices': '/'.join(policy.devices),
                    'n_gpu_layers': policy.gpu_layers, 'cpu_mask': '0x0', 'cpu_strict': False,
                    'poll': 50, 'main_gpu': 0}
        for key, value in expected.items():
            if type(row.get(key)) is not type(value) or row[key] != value:
                raise ValueError('diagnostic '+key+' mismatch')
        for key in ('n_prompt', 'n_gen', 'n_depth', 'n_batch', 'n_ubatch', 'n_threads',
                    'fit_target', 'fit_min_ctx', 'flash_attn', 'no_op_offload', 'n_cpu_moe'):
            if type(row.get(key)) is not int:
                raise ValueError('non-integer benchmark control: '+key)
        for key in ('no_kv_offload', 'no_host', 'repack', 'embeddings'):
            if type(row.get(key)) is not bool:
                raise ValueError('non-boolean benchmark control: '+key)
        commit = row.get('build_commit', '')
        if not re.fullmatch('[0-9a-f]{7,40}', commit) or not config.source_revision.startswith(commit):
            raise ValueError('benchmark build commit mismatch')
        ratios = [float(v) for v in row.get('tensor_split', '').split('/')]
        expected_ratio = list(policy.tensor_split) if policy.tensor_split else [0.0]
        if any(not math.isfinite(v) for v in ratios) or ratios != expected_ratio:
            raise ValueError('benchmark split request mismatch')
    return [{'n_prompt': r['n_prompt'], 'n_gen': r['n_gen'], 'n_depth': r['n_depth'],
             'samples_ns': r['samples_ns']} for r in rows]


def validate_diagnostic(config, policy, stdout, stderr):
    evidence = placement(stderr)
    errors = []
    workloads = []
    try:
        workloads = validate_rows(stdout, config, policy)
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        errors.append(str(e))
    layout = read_layout(config.model)
    errors += verify_placement(evidence, policy, layout)
    pre = json.loads((Path(config.output)/'preflight.json').read_text())
    predicted = {str(i): d for d, ids in pre['screening']['estimated_layers'].items() for i in ids}
    predicted.update({str(i): 'CPU' for i in range(layout.blocks+1) if str(i) not in predicted})
    if policy.tensor_split is not None or len(policy.devices) == 1:
        if evidence['layers'] != predicted:
            errors.append('actual mapping differs from fixed requested policy')
    else:
        # Default is a runtime free-VRAM decision; record differences from prediction.
        cuda_count = sum(d == 'CUDA0' for d in evidence['layers'].values())
        expected = {str(i): 'CUDA0' if i < cuda_count else 'ROCm0' for i in range(layout.blocks+1)}
        if not 0 < cuda_count < layout.blocks+1 or evidence['layers'] != expected:
            errors.append('default mixed map is not a complete contiguous two-GPU split')
    for device in policy.devices:
        identities = re.findall(r'using device '+device+r' \([^\n]*?\) \(([^)]+)\)', stderr)
        if not identities or set(identities) != {config.devices[device]}:
            errors.append(device+': runtime PCI evidence missing or mismatched')
        kv = re.findall(r'\b'+device+r'\s+KV buffer size\s*=\s*([\d.]+) MiB', stderr)
        if not kv or any(float(v) <= 0 for v in kv):
            errors.append(device+': positive KV allocation missing')
    if set(re.findall(r'using device ((?:CUDA|ROCm)\d+) \(', stderr)) != set(policy.devices):
        errors.append('actual selected devices differ from request')
    if not {512, 256} <= set(evidence['contexts']):
        errors.append('PP512 / padded TG256 contexts missing')
    if not re.search(r'\(load_mode\s*=\s*none\)', stderr):
        errors.append('explicit non-mmap loading evidence missing')
    buffers = re.findall(r'(\w+)\s+model buffer size\s*=\s*([\d.]+) MiB', stderr)
    if len(policy.devices) == 1:
        if 'CPU' not in evidence['matrix_compute_backends']:
            errors.append('CPU-offloaded matrix scheduling missing')
        if not any((d.startswith('CPU') or d.endswith('_Host')) and float(v)>0 for d, v in buffers):
            errors.append('resident CPU/host model allocation missing')
    evidence['all_model_buffer_lines'] = [s for s in stderr.splitlines() if 'model buffer size' in s]
    evidence['all_compute_buffer_lines'] = [s for s in stderr.splitlines() if 'compute buffer size' in s]
    evidence['matches_preflight_prediction'] = evidence['layers'] == predicted
    return {'errors': errors, 'placement': evidence, 'diagnostic_workloads': workloads,
            'timings_eligible_for_comparison': False,
            'semantic_answer_test': False}


def resource_summary(directory):
    directory = Path(directory)
    def records(name):
        path = directory/name
        return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    fast = [r for r in records('fast.jsonl') if r.get('data') and not r.get('error')]
    gpu = [r for r in records('gpu.jsonl') if r.get('data') and not r.get('error')]
    summary = {'scope': 'C0 diagnostics, not formal throughput; peaks sampled except wait4 RSS',
               'fast_samples': len(fast), 'gpu_samples': len(gpu)}
    if fast:
        summary.update(min_available_bytes=min(r['data']['memory']['MemAvailable'] for r in fast),
                       peak_swap_used_bytes=max(r['data']['swap_used'] for r in fast),
                       peak_sampled_rss_bytes=max(r['data']['process']['VmRSS'] for r in fast),
                       peak_sampled_hwm_bytes=max(r['data']['process']['VmHWM'] for r in fast))
    if gpu:
        summary['gpu'] = {d: {'peak_used_bytes': max(r['data'][d]['used_bytes'] for r in gpu),
                              'peak_temperature_c': max(r['data'][d]['temperature_c'] for r in gpu)}
                          for d in ('CUDA0', 'ROCm0')}
    thermal = directory/'thermal-summary.json'
    if thermal.exists():
        summary['host_thermal'] = json.loads(thermal.read_text())
    path = directory/'wait4-rusage.json'
    if path.exists():
        summary['wait4'] = json.loads(path.read_text())
    return summary


def mixed_comparison(results):
    by_name = {r['policy']: r['result'] for r in results}
    names = ('mixed-equal', 'mixed-default')
    if any(n not in by_name or by_name[n]['status'] != 'passed' for n in names):
        return {'status': 'unavailable', 'reason': 'both mixed diagnostics must pass'}
    equal, default = [by_name[n]['validation']['placement'] for n in names]
    changed = [i for i in equal['layers'] if equal['layers'][i] != default['layers'][i]]
    return {'status': 'observed', 'same_layer_mapping': not changed,
            'changed_layers': sorted(changed, key=int),
            'equal_model_buffers_mib': equal['model_buffers_mib'],
            'default_model_buffers_mib': default['model_buffers_mib'],
            'interpretation': ('No distinct layer placement observed' if not changed else
                               'Distinct layer placement observed; no speed conclusion from diagnostics')}


def execute(config, plan):
    """Four attempts maximum; any failure stops. Caller owns the execution lock."""
    root = Path(config.output)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + SUITE_SECONDS
    interrupted = []
    previous = {s: signal.signal(s, lambda sig, frame: interrupted.append(sig))
                for s in (signal.SIGTERM, signal.SIGINT)}
    result = {'stage': 'C0', 'status': 'incomplete', 'attempts': [], 'launches': 0,
              'timings_eligible_for_comparison': False, 'started': stamp()}
    def check():
        if interrupted:
            raise InterruptedError('suite_signal')
        if time.monotonic() >= deadline:
            raise TimeoutError('suite_deadline')
    try:
        atomic_json(root/'suite-result.json', result)
        atomic_json(root/'plan.json', plan)
        atomic_json(root/'preview.json', preview(config))
        atomic_json(root/'code-identities.json', {str(p): file_identity(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        memory = parse_meminfo(Path('/proc/meminfo').read_text())
        baseline = memory['SwapTotal']-memory['SwapFree']
        atomic_json(root/'suite-baseline.json', {'swap_used': baseline, 'memory': memory, **stamp()})
        for ordinal, policy in enumerate(POLICIES, 1):
            check()
            if ordinal > 1:
                append_json(root/'suite-events.jsonl', {'event': 'cooldown', **stamp()})
                until = time.monotonic()+COOLDOWN_SECONDS
                while time.monotonic() < until:
                    check(); time.sleep(.1)
            check()
            if deadline-time.monotonic() < C0_LIMITS.overall:
                raise TimeoutError('insufficient suite time for a full launch budget')
            child = replace(config, output=str(root/f'{ordinal:02d}-{policy.name}'))
            append_json(root/'suite-events.jsonl', {'event': 'diagnostic_attempt', 'policy': policy.name, **stamp()})
            run = supervise(command(child, policy), child.output, environment(child),
                            partial(preflight, child, plan, policy, baseline),
                            partial(sampler_factory, child), partial(validate_diagnostic, child, policy),
                            C0_LIMITS, scope='one C0 diagnostic benchmark; no retry; timings excluded from comparison')
            result['attempts'].append({'ordinal': ordinal, 'policy': policy.name, 'directory': child.output, 'result': run})
            result['launches'] += run['launches']
            atomic_json(root/'suite-result.json', result)
            atomic_json(Path(child.output)/'resource-summary.json', resource_summary(child.output))
            if run['status'] != 'passed':
                raise RuntimeError(policy.name+': '+str(run['reason']))
        check()
        result['status'] = 'passed'
    except BaseException as e:
        result.update(status='failed', error=f'{type(e).__name__}: {e}')
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        result['received_signals'] = interrupted
        result['ended'] = stamp()
        result['elapsed_seconds'] = time.monotonic()-started
        result['mixed_comparison'] = mixed_comparison(result['attempts'])
        atomic_json(root/'suite-result.json', result)
        atomic_json(root/'mixed-comparison.json', result['mixed_comparison'])
        append_json(root/'suite-events.jsonl', {'event': 'final', 'status': result['status'], **stamp()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--execute', action='store_true', help='Requires separate explicit Stage C0 approval')
    args = parser.parse_args(argv)
    try:
        config, plan = load_plan(args.plan)
        if not args.execute:
            print(json.dumps(preview(config), indent=2))
            return 0
        with execution_lock():
            result = execute(config, plan)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'passed' else 2
    except (OSError, ValueError) as e:
        print(json.dumps({'status': 'refused', 'error': str(e)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
