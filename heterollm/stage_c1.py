"""C1: twelve independent quiet launches; preview only unless explicitly approved."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
from functools import partial
import json
import os
from pathlib import Path
import re
import signal
import statistics
import time
import traceback

from . import stage_c as c0
from .analysis import aggregate, parse_benchmark
from .config import digest
from .gpu_sensors import problems
from .hardware import parse_meminfo, sha256_file
from .pilot import load_plan as load_pilot_plan, sampler_factory
from .runner import execution_lock
from .supervisor import Reader, atomic_json, append_json, evidence_sizes, guard, stamp, supervise

MIB = 1024**2
GIB = 1024**3
ROOT = c0.ROOT
LIMITS = c0.C0_LIMITS
SUITE_SECONDS = 11400
REPS = 3
ORDER = (('E','D','A','B'), ('D','B','E','A'), ('B','A','D','E'))
BY_ID = dict(zip('EDAB', c0.POLICIES))
VRAM_TOLERANCE = 128*MIB
PEAK_ALLOWANCE = 128*MIB
MIN_MATCHING_SAMPLES = 3
MIN_MATCHING_SPAN = 2
COOLDOWN_MAX_SECONDS = 600
COOLDOWN_STABLE_SECONDS = 10
COOLDOWN_STABLE_SAMPLES = 11


def command(config, policy):
    argv = c0.command(config, policy)
    argv.remove('-v')
    argv[argv.index('-r')+1] = str(REPS)
    return argv


def environment(config):
    env = c0.environment(config)
    del env['GGML_SCHED_DEBUG']
    return env


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted((ROOT/'heterollm').glob('*.py'))}


def manifest(config, plan):
    launches = []
    for round_number, order in enumerate(ORDER, 1):
        for label in order:
            ordinal = len(launches)+1
            policy = BY_ID[label]
            launches.append({'ordinal': ordinal, 'round': round_number, 'id': label,
                             'policy': policy.to_dict(), 'argv': command(config, policy),
                             'environment': environment(config),
                             'output': str(Path(config.output)/f'{ordinal:02d}-r{round_number}-{policy.name}')})
    # JSON normalization makes saved/read tuples identical; this is also the
    # exact manifest checked at execution, not just a descriptive preview.
    value = {'stage': 'C1', 'launches': launches, 'maximum_launches': 12,
             'limits': asdict(LIMITS), 'suite_seconds': SUITE_SECONDS,
             'cooldown_seconds': c0.COOLDOWN_SECONDS, 'internal_repetitions': REPS,
             'cooldown_protocol': {'before_every_launch': True,
                                   'minimum_seconds': c0.COOLDOWN_SECONDS,
                                   'maximum_seconds': COOLDOWN_MAX_SECONDS,
                                   'stable_seconds': COOLDOWN_STABLE_SECONDS,
                                   'minimum_consecutive_gpu_samples': COOLDOWN_STABLE_SAMPLES,
                                   'temperature_ceiling_c': LIMITS.admission_temperature,
                                   'gpu_sample_period_seconds': 1,
                                   'next_attempt_reserved_seconds': LIMITS.overall},
             'independent_observation': 'one new process, summarized by its three repetitions',
             'c0_reference_sha256': plan['reference_sha256'], 'source_sha256': source_hashes(),
             'placement_contract': {'plateau_tolerance_bytes': VRAM_TOLERANCE,
                                    'peak_allowance_bytes': PEAK_ALLOWANCE,
                                    'matching_samples': MIN_MATCHING_SAMPLES,
                                    'matching_span_seconds': MIN_MATCHING_SPAN,
                                    'fresh_layer_trace': False}}
    return json.loads(json.dumps(value))


def load_plan(path):
    config, plan = load_pilot_plan(path)
    required = {'stage','config','receipt','runtime_baseline','dependency_baseline',
                'hardware_baseline','mem_total_bytes','authorization','reference',
                'reference_sha256','manifest'}
    if set(plan) != required or plan['stage'] != 'C1' or plan['mem_total_bytes'] != 98774167552:
        raise ValueError('only the reviewed twelve-launch C1 plan is supported')
    paths = {'source': ROOT/'llama.cpp', 'bench': ROOT/'llama.cpp/build-mixed/bin/llama-bench',
             'cli': ROOT/'llama.cpp/build-mixed/bin/llama-cli',
             'model': ROOT/'models/llama33-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf'}
    if any(Path(getattr(config,k)) != v for k,v in paths.items()):
        raise ValueError('unreviewed model/runtime paths')
    if (config.max_launches,config.timeout_seconds,config.suite_seconds,
        config.max_swap_growth_mib,config.minimum_available_ram_mib,
        config.max_temperature_c,config.max_log_mib) != (12,900,11400,256,4096,85,256):
        raise ValueError('unreviewed C1 execution controls')
    out = Path(config.output)
    if out.parent != ROOT/'runs/llama33-70b' or not out.name.startswith('stage-c1-'):
        raise ValueError('C1 requires a dedicated fresh output path')
    if sha256_file(plan['reference']) != plan['reference_sha256']:
        raise ValueError('C0 reference file changed')
    if c0.strict_json(Path(plan['manifest']).read_text()) != manifest(config,plan):
        raise ValueError('exact reviewed C1 manifest/source identity changed')
    return config, plan


def read_records(path):
    return [c0.strict_json(s) for s in Path(path).read_text().splitlines()]


def active_samples(directory):
    """Use the child-owned exec/reap interval and the pre-exec sensor sample."""
    directory = Path(directory)
    events = read_records(directory/'events.jsonl')
    release = [e['monotonic'] for e in events if e['event']=='exec_release']
    reap = [e['monotonic'] for e in events if e['event']=='reaped']
    if len(release)!=1 or len(reap)!=1 or not release[0]<reap[0]:
        raise ValueError('missing or ambiguous child execution interval')
    records = {kind:read_records(directory/(kind+'.jsonl')) for kind in ('gpu','fast')}
    for kind, rows in records.items():
        if any(r.get('error') or not r.get('data') for r in rows):
            raise ValueError(kind+': rejected resource sample')
    before = [r for r in records['gpu'] if r['ended']['monotonic']<=release[0]]
    if not before:
        raise ValueError('missing independent pre-exec GPU baseline')
    baseline = max(before,key=lambda r:r['ended']['monotonic'])
    if release[0]-baseline['ended']['monotonic']>3:
        raise ValueError('stale pre-exec GPU baseline')
    active = {kind:[r for r in rows if r['started']['monotonic']>=release[0]
                   and r['ended']['monotonic']<=reap[0]] for kind,rows in records.items()}
    if len(active['gpu'])<MIN_MATCHING_SAMPLES or len(active['fast'])<2:
        raise ValueError('insufficient independent resource observations')
    return baseline['data'], active


def calibration(directory, policy):
    """Calibrate from preserved C0 observations, never from a C1 result."""
    directory = Path(directory)
    result = c0.strict_json((directory/'result.json').read_text())
    if (result['status']!='passed' or result['exit_code']!=0 or result['launches']!=1 or
        not result['cleanup_complete'] or not result['reader_cleanup_complete'] or
        result['validation']['errors'] or result['validation']['timings_eligible_for_comparison']):
        raise ValueError('C0 reference was not a successful diagnostic')
    pre = c0.strict_json((directory/'preflight.json').read_text())
    if pre['policy'] != json.loads(json.dumps(policy.to_dict())):
        raise ValueError('C0 policy mismatch')
    placement = result['validation']['placement']
    baseline, active = active_samples(directory)
    floor = {d:placement['model_buffers_mib'].get(d,0)*MIB for d in policy.devices}
    full = [r for r in active['gpu'] if all(
        r['data'][d]['used_bytes']-baseline[d]['used_bytes']>=floor[d] for d in policy.devices)]
    if len(full)<MIN_MATCHING_SAMPLES:
        raise ValueError('insufficient C0 full-allocation observations')
    profile = {}
    for d in ('CUDA0','ROCm0'):
        deltas = [r['data'][d]['used_bytes']-baseline[d]['used_bytes'] for r in full]
        profile[d] = {'median_delta_bytes': statistics.median(deltas),
                      'peak_delta_bytes': max(r['data'][d]['used_bytes']-baseline[d]['used_bytes']
                                              for r in active['gpu'])}
    peak_rss = max(r['data']['process']['VmHWM'] for r in active['fast'])
    return {'policy': json.loads(json.dumps(policy.to_dict())), 'layers': placement['layers'],
            'c0_model_buffers_mib': placement['model_buffers_mib'], 'gpu': profile,
            'c0_peak_sampled_hwm_bytes': peak_rss,
            'host_estimated_bytes': pre['screening']['host_estimated_bytes'],
            'baseline': baseline, 'full_allocation_samples': len(full)}


def verify_reference(plan):
    reference = c0.strict_json(Path(plan['reference']).read_text())
    if sha256_file(plan['reference']) != plan['reference_sha256']:
        raise ValueError('reference digest mismatch')
    root = Path(reference['root'])
    if root != ROOT/'runs/llama33-70b/stage-c0-diagnostics-20261020':
        raise ValueError('unexpected C0 evidence root')
    if not reference['files']:
        raise ValueError('empty C0 evidence manifest')
    for name, record in reference['files'].items():
        path = root/name
        if path.resolve().parent != root.resolve() and root.resolve() not in path.resolve().parents:
            raise ValueError('C0 evidence path escapes root')
        if path.stat().st_size!=record['bytes'] or sha256_file(path)!=record['sha256']:
            raise ValueError('preserved C0 evidence changed: '+name)
    suite = c0.strict_json((root/'suite-result.json').read_text())
    if suite['status']!='passed' or suite['launches']!=4:
        raise ValueError('C0 suite incomplete')
    for label, policy in BY_ID.items():
        profile = calibration(root/reference['directories'][label],policy)
        if profile != reference['profiles'][label]:
            raise ValueError('C0 calibration differs from reviewed reference')
    return reference


def preflight(config,plan,label,swap_baseline):
    reference = verify_reference(plan)
    result = c0.preflight(config,plan,BY_ID[label],swap_baseline)
    profile = reference['profiles'][label]
    predicted = {str(i):d for d,ids in result['screening']['estimated_layers'].items() for i in ids}
    predicted.update({str(i):'CPU' for i in range(81) if str(i) not in predicted})
    if predicted != profile['layers']:
        raise ValueError('predicted placement changed from C0; review required')
    result['c0_reference'] = {'sha256':plan['reference_sha256'],'id':label,
                              'profile':profile,'fresh_per_layer_trace':False}
    return result


def allocation_check(directory, policy, profile):
    baseline, active = active_samples(directory)
    matched = []
    observed = {}
    for d in ('CUDA0','ROCm0'):
        deltas = [r['data'][d]['used_bytes']-baseline[d]['used_bytes'] for r in active['gpu']]
        peak = max(deltas)
        expected = profile['gpu'][d]
        if peak>expected['peak_delta_bytes']+PEAK_ALLOWANCE:
            raise ValueError(d+': unexpected VRAM growth/background interference')
        observed[d] = {'baseline_used_bytes':baseline[d]['used_bytes'],
                       'peak_delta_bytes':peak,'c0_median_delta_bytes':expected['median_delta_bytes'],
                       'c0_peak_delta_bytes':expected['peak_delta_bytes']}
    for row in active['gpu']:
        if all(abs(row['data'][d]['used_bytes']-baseline[d]['used_bytes']-
                   profile['gpu'][d]['median_delta_bytes'])<=VRAM_TOLERANCE
               for d in ('CUDA0','ROCm0')):
            matched.append(row)
    if (len(matched)<MIN_MATCHING_SAMPLES or
        matched[-1]['started']['monotonic']-matched[0]['started']['monotonic']<MIN_MATCHING_SPAN):
        raise ValueError('no sustained C0-consistent two-device allocation pattern')
    for d in policy.devices:
        activity = [r['data'][d].get('utilization_percent',r['data'][d].get('busy_percent')) for r in matched]
        if not any(type(v) in (int,float) and 0<v<=100 for v in activity):
            raise ValueError(d+': no independent activity observation during matching allocation')
    fast = [r['data']['process'] for r in active['fast']]
    peak_rss = max(r['VmHWM'] for r in fast)
    # Bound variation around the observed C0 process footprint; the 32 GiB
    # safety ceiling is separate and unchanged. Partial/misplaced CPU offload
    # must not silently qualify as the reviewed baseline.
    if abs(peak_rss-profile['c0_peak_sampled_hwm_bytes'])>GIB:
        raise ValueError('process RSS/HWM differs from C0 by more than 1 GiB')
    if peak_rss<profile['host_estimated_bytes']:
        raise ValueError('resident host model/KV footprint not established')
    user_ticks = max(r['user_ticks'] for r in fast)-min(r['user_ticks'] for r in fast)
    if len(policy.devices)==1 and user_ticks<os.sysconf('SC_CLK_TCK'):
        raise ValueError('insufficient process CPU activity for offloaded configuration')
    return {'status':'allocation_consistent_with_C0','observed_gpu':observed,
            'matching_samples':len(matched),'matching_span_seconds':
            matched[-1]['started']['monotonic']-matched[0]['started']['monotonic'],
            'peak_sampled_hwm_bytes':peak_rss,'process_user_ticks_delta':user_ticks,
            'c0_observed_layer_map':profile['layers'],'fresh_observed_layer_map':None,
            'limitation':'System-wide GPU allocation/activity corroborates C0, but is not a fresh per-layer or per-kernel trace; background allocation can confound it.'}


def validate(config,policy,stdout,stderr):
    errors = []
    metrics = None
    proof = None
    try:
        c0.validate_rows(stdout,config,policy,reps=REPS)
        metrics = parse_benchmark(c0.strict_json(stdout),config,REPS,load_mode='none')
        if re.search(r'node #|assigned to device|ggml_backend_sched_print_assignments',stderr):
            raise ValueError('verbose placement/scheduler output contaminates timed run')
        pre = c0.strict_json((Path(config.output)/'preflight.json').read_text())
        proof = allocation_check(config.output,policy,pre['c0_reference']['profile'])
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as e:
        errors.append(str(e))
    return {'errors':errors,'metrics':metrics,'placement_verification':proof,
            'timings_eligible_for_comparison':not errors,'internal_repetitions':REPS,
            'independent_observation':'one process','semantic_answer_test':False}


def summarize(records):
    if (len(records)!=12 or len({r['run_id'] for r in records})!=12 or
        Counter((r['policy']['name'],r['round']) for r in records)!=
        Counter((p.name,r) for p in BY_ID.values() for r in (1,2,3))):
        raise ValueError('complete twelve independent launches required for comparison')
    groups = aggregate(records,phase='c1')
    if set(groups)!={p.name for p in BY_ID.values()} or any(g['pp']['n']!=3 or g['tg']['n']!=3 for g in groups.values()):
        raise ValueError('three independent observations per configuration required')
    return {'independent_n_per_configuration':3,'groups':groups,'includes_C0':False,
            'within_process_repetitions_are_independent':False,
            'interpretation':'Initial descriptive comparison only; small n and ordering effects remain. No optimizer selection or general speedup claim.'}


def cooldown(config, suite_deadline, swap_baseline, check):
    """Bounded thermal gate; only existing sensor/proc readers, never a model."""
    directory = Path(config.output)
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {'status':'incomplete', 'started':stamp(), 'reader_cleanup_complete':False,
              'stable_samples':0, 'stable_seconds':0, 'swap_baseline':swap_baseline}
    readers = []; latest = {}; stable_start = None; oom_baseline = None
    identity = None

    def budget():
        check()
        now = time.monotonic()
        if now >= suite_deadline - LIMITS.overall:
            raise TimeoutError('cooldown: insufficient suite time for a full launch budget')
        if now - start >= COOLDOWN_MAX_SECONDS:
            raise TimeoutError('cooldown: temperature readiness deadline')
        return now

    try:
        atomic_json(directory/'result.json', result)
        budget()
        funcs = sampler_factory(config, os.getpid())
        for name,period,limit in [('fast',.25,1),('gpu',1,2)]:
            readers.append(Reader(name, funcs[name], directory, period, limit))
        while True:
            budget()
            for reader in readers:
                value = reader.poll()
                if value:
                    latest[reader.name] = value
                    data = value['data']
                    if reader.name == 'fast':
                        current = data['process']['start_ticks']
                        if identity is not None and current != identity:
                            raise RuntimeError('cooldown: process identity changed')
                        identity = current
                        if oom_baseline is not None and data['oom_kill'] > oom_baseline:
                            raise RuntimeError('cooldown: system OOM kill counter increased')
                        oom_baseline = data['oom_kill']
                    else:
                        if set(data) != set(config.devices):
                            raise ValueError('cooldown: required GPU missing')
                        for device,pci in config.devices.items():
                            if data[device].get('pci') != pci or any(problems(data[device]).values()):
                                raise ValueError('cooldown: invalid GPU identity or mandatory sensor: '+device)
                        if all(g['temperature_c'] <= LIMITS.admission_temperature for g in data.values()):
                            if stable_start is None:
                                stable_start = value['ended']['monotonic']
                            result['stable_samples'] += 1
                            result['stable_seconds'] = value['ended']['monotonic'] - stable_start
                        else:
                            stable_start = None
                            result.update(stable_samples=0, stable_seconds=0)
                        result['last_gpu_reading'] = value
                if reader.name in latest and time.monotonic()-latest[reader.name]['started']['monotonic'] > {'fast':1,'gpu':3}[reader.name]:
                    raise RuntimeError('cooldown: '+reader.name+' stale sample')
            now = budget()
            if sum(evidence_sizes(directory).values()) > LIMITS.evidence_cap:
                raise RuntimeError('cooldown: evidence_limit')
            if len(latest) == 2:
                # No model exists during cooldown. The process counters refer
                # to this recorder; child safeguards remain in supervise().
                reason = guard(latest['fast']['data'], latest['gpu']['data'],
                               {'model_mappings':[]}, swap_baseline, LIMITS)
                if reason:
                    raise RuntimeError('cooldown: '+reason)
                if (now-start >= c0.COOLDOWN_SECONDS and
                    result['stable_samples'] >= COOLDOWN_STABLE_SAMPLES and
                    result['stable_seconds'] >= COOLDOWN_STABLE_SECONDS):
                    result['status'] = 'ready'
                    break
            time.sleep(.02)
    except BaseException as e:
        result.update(status='failed', error=f'{type(e).__name__}: {e}',
                      traceback=traceback.format_exc())
    finally:
        cleanup = []
        for reader in readers:
            try:
                cleanup.append({'name':reader.name, 'pid':reader.process.pid, 'closed':reader.close()})
            except BaseException as e:
                cleanup.append({'name':reader.name, 'closed':False, 'error':str(e),
                                'traceback':traceback.format_exc()})
        complete = all(item['closed'] for item in cleanup)
        result.update(reader_cleanup=cleanup, reader_cleanup_complete=complete,
                      ended=stamp(), elapsed_seconds=time.monotonic()-start)
        if not complete:
            result.update(status='failed', cleanup_error='cooldown reader cleanup incomplete')
        atomic_json(directory/'result.json', result)
    return result


def execute(config,plan):
    directory = Path(config.output)
    directory.mkdir(parents=True,exist_ok=False)
    start = time.monotonic();deadline = start+SUITE_SECONDS
    interrupted = []
    previous = {s:signal.signal(s,lambda sig,frame:interrupted.append(sig)) for s in (signal.SIGTERM,signal.SIGINT)}
    result = {'stage':'C1','status':'incomplete','launches':0,'attempts':[],'measurements':[], 'cooldowns':[],
              'comparison':None,'started':stamp()}
    def check():
        if interrupted:raise InterruptedError('suite_signal')
        if time.monotonic()>=deadline:raise TimeoutError('suite_deadline')
    try:
        atomic_json(directory/'suite-result.json',result)
        exact = manifest(config,plan)
        atomic_json(directory/'manifest.json',exact)
        atomic_json(directory/'plan.json',plan)
        memory = parse_meminfo(Path('/proc/meminfo').read_text())
        swap = memory['SwapTotal']-memory['SwapFree']
        atomic_json(directory/'suite-baseline.json',{'memory':memory,'swap_used':swap,**stamp()})
        comparison_key = digest({'manifest':exact,'hardware_baseline':sha256_file(plan['hardware_baseline']),
                                 'model_sha256':config.model_sha256,'runtime':sha256_file(plan['runtime_baseline'])})
        for launch in exact['launches']:
            check()
            thermal = replace(config,output=str(directory/f"cooldown-{launch['ordinal']:02d}-{launch['id']}"))
            append_json(directory/'suite-events.jsonl',{'event':'cooldown','ordinal':launch['ordinal'],**stamp()})
            ready = cooldown(thermal,deadline,swap,check)
            result['cooldowns'].append({'ordinal':launch['ordinal'],'id':launch['id'],
                                        'directory':thermal.output,'result':ready})
            atomic_json(directory/'suite-result.json',result)
            if ready['status']!='ready':
                raise RuntimeError('cooldown failed: '+str(ready.get('error',ready.get('cleanup_error'))))
            check()
            if deadline-time.monotonic()<LIMITS.overall:
                raise TimeoutError('insufficient suite time for a full launch budget')
            policy = BY_ID[launch['id']]
            child = replace(config,output=launch['output'])
            append_json(directory/'suite-events.jsonl',{'event':'timed_attempt','ordinal':launch['ordinal'],'id':launch['id'],**stamp()})
            run = supervise(launch['argv'],child.output,launch['environment'],
                            partial(preflight,child,plan,launch['id'],swap),
                            partial(sampler_factory,child),partial(validate,child,policy),
                            LIMITS,scope='one C1 timed process; no retry or extra diagnostics')
            result['attempts'].append({'ordinal':launch['ordinal'],'id':launch['id'],
                                       'round':launch['round'],'directory':child.output,'result':run})
            result['launches'] += run['launches']
            atomic_json(directory/'suite-result.json',result)
            resources = c0.resource_summary(child.output)
            resources['scope'] = 'C1 resources: sampled peaks except wait4 lifetime RSS'
            atomic_json(Path(child.output)/'resource-summary.json',resources)
            if run['status']!='passed':
                raise RuntimeError(launch['id']+': '+str(run['reason']))
            record = {'run_id':launch['output'],'round':launch['round'],'policy':launch['policy'],
                      'phase':'c1','status':'ok','comparison_key':comparison_key,
                      'validation_key':digest({'reference':plan['reference_sha256'],'policy':policy.to_dict()}),
                      'metrics':run['validation']['metrics']}
            result['measurements'].append(record)
            atomic_json(Path(child.output)/'measurement.json',record)
            atomic_json(directory/'suite-result.json',result)
        check()
        result['comparison'] = summarize(result['measurements'])
        result['status'] = 'passed'
    except BaseException as e:
        result.update(status='failed',error=f'{type(e).__name__}: {e}',
                      traceback=traceback.format_exc())
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
        result.update(received_signals=interrupted,ended=stamp(),elapsed_seconds=time.monotonic()-start)
        atomic_json(directory/'suite-result.json',result)
        append_json(directory/'suite-events.jsonl',{'event':'final','status':result['status'],**stamp()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--execute',action='store_true',help='Requires separate explicit C1 approval')
    args = parser.parse_args(argv)
    try:
        config,plan = load_plan(args.plan)
        if not args.execute:
            print(json.dumps({'mode':'preview_only','output_exists':Path(config.output).exists(),
                              'manifest':manifest(config,plan)},indent=2))
            return 0
        with execution_lock():
            result = execute(config,plan)
        print(json.dumps(result,indent=2))
        return 0 if result['status']=='passed' else 2
    except (OSError,ValueError) as e:
        print(json.dumps({'status':'refused','error':str(e)}))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
