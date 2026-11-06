"""Separate U/T overhead and numerical contracts. Preview is CPU-only by default."""
import argparse
from dataclasses import asdict,replace
from functools import partial
import json
import math
from pathlib import Path
import signal
import time
import traceback

from . import stage_c as c0, stage_c1 as c1
from .stage_c2_timing import adapter as timing
from . import calibration_analysis as analysis
from .config import Config,digest
from .hardware import file_identity,parse_meminfo,query,sha256_file,clean_environment
from .pilot import sampler_factory
from .runner import execution_lock
from .supervisor import atomic_json,append_json,stamp,supervise
from .timing_analysis import analyze_timing

ROOT=c0.ROOT
PREP=ROOT/'runs/llama33-70b/c2-calibration-equivalence-prep-20261027'
FIXTURE=ROOT/'instrumentation/calibration/tokens.txt'
HARNESS_RECEIPT=ROOT/'configs/stage-c2-equivalence-build-receipt.json'
POLICY=c0.POLICIES[0]
LIMITS=c0.C0_LIMITS
TOTALS={'overhead':11400,'equivalence':3000}
BASE_PLANS={'U':ROOT/'configs/stage-c1-memtotal-fixed-plan.json','T':ROOT/'configs/stage-c2-timing-plan.json'}

def expected_plan(kind):
    if kind not in TOTALS:raise ValueError('unsupported calibration kind')
    return {'schema':'heterollm.calibration.plan.v1','kind':kind,
        'output':str(ROOT/f'runs/llama33-70b/stage-c2-{kind}-20261027'),
        'manifest':str(ROOT/f'configs/stage-c2-{kind}-manifest.json'),
        'authorization':'PREPARATION ONLY; independent review and separate explicit owner GPU authorization required for this experiment.'}

def base_plan(arm):return c0.strict_json(BASE_PLANS[arm].read_text())
def configuration(arm,output):
    base=Config.load(base_plan(arm)['config'])
    return replace(base,output=str(output))

def build_dependencies(exe,config):
    result=query(['ldd',str(exe)],clean_environment(config.environment),timeout=5)
    if result['exit_code'] or 'not found' in result['stdout']:raise ValueError('unresolved correctness-harness library')
    import re
    paths=[Path(s).resolve() for s in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(',result['stdout'],re.M)]
    binpath=Path(config.bench).parent
    if any(p.name.startswith(('libggml','libllama')) and p.parent!=binpath for p in paths):
        raise ValueError('correctness harness resolved wrong runtime')
    if not any(p.name.startswith('libllama') for p in paths):raise ValueError('harness libllama missing')
    return {'executable':file_identity(exe),'dependencies':{str(p):file_identity(p) for p in sorted(set(paths))},'ldd':result}

def harness_path(arm):return PREP/'bin'/arm/'equivalence-capture'

def manifest(plan):
    kind=plan['kind'];pairs=analysis.ORDERS if kind=='overhead' else ('UT',)
    launches=[]
    for block,order in enumerate(pairs,1):
        for arm in order:
            ordinal=len(launches)+1;out=Path(plan['output'])/f'{ordinal:02d}-block{block}-{arm}'
            conf=configuration(arm,out)
            env=timing.environment(conf) if arm=='T' else c1.environment(conf)
            argv=c1.command(conf,POLICY) if kind=='overhead' else [str(harness_path(arm)),arm,conf.model,str(FIXTURE),str(out),str(Path(conf.bench).parent)]
            launches.append({'ordinal':ordinal,'block':block,'arm':arm,'output':str(out),'argv':argv,'environment':env,
                             'configuration':asdict(conf)})
    bound={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'heterollm').glob('*.py'))}
    paths={*BASE_PLANS.values(),ROOT/'instrumentation/v3/phase-timing.patch',FIXTURE}
    for p in BASE_PLANS.values():
        baseline=c0.strict_json(p.read_text())
        paths.update(Path(baseline[k]) for k in ('config','receipt','hardware_baseline','reference'))
        paths.update(Path(baseline[k]) for k in ('runtime_baseline','dependency_baseline','runtime_receipt') if k in baseline)
    paths.update((ROOT/'instrumentation/calibration').glob('*.cpp'))
    paths.update((ROOT/'tests').glob('test_calibration*.py'))
    paths.update((ROOT/'tests/fixtures').glob('calibration*.cpp'))
    paths.add(ROOT/'scripts/build-c2-equivalence.py')
    paths.update(ROOT/'docs'/name for name in ('C2_TIMING_OVERHEAD_METHODOLOGY.md','C2_OVERHEAD_EXECUTION_PROPOSAL.md','C2_NUMERICAL_EQUIVALENCE_PROPOSAL.md'))
    if kind=='equivalence':paths.add(HARNESS_RECEIPT)
    for p in sorted(paths):bound[str(p.relative_to(ROOT))]=sha256_file(p)
    return {'schema':'heterollm.calibration.manifest.v1','plan_sha256':digest(plan),'kind':kind,'launches':launches,
        'maximum_model_processes':len(launches),'suite_seconds':TOTALS[kind],'limits':asdict(LIMITS),
        'internal_repetitions':3 if kind=='overhead' else 1,'source_and_receipt_sha256':bound,
        'cooldown':{'before_every_process':True,'minimum_seconds':30,'maximum_seconds':600,'stable_seconds':10,'fresh_samples':11,'max_c':45},
        'host_thermal':{'policy':timing.host_thermal.POLICY,'sustained_seconds':25,'admission_max_seconds':120},
        'placement_reference':{'label':'E','plateau_tolerance_bytes':c1.VRAM_TOLERANCE,'peak_allowance_bytes':c1.PEAK_ALLOWANCE,
            'matching_samples':c1.MIN_MATCHING_SAMPLES,'matching_span_seconds':c1.MIN_MATCHING_SPAN,'fresh_layer_trace':False},
        'trace_caps_bytes':{'cpu':64*1024**2,'device':32*1024**2},
        'numerical_contract':None if kind=='overhead' else {'rows':analysis.ROWS,'vocabulary':analysis.VOCAB,'atol':analysis.ATOL,'rtol':analysis.RTOL,
            'formula':'abs(T-U) <= atol + rtol*abs(U), every value; exact argmax and piece bytes','teacher_forced':True,'input_sha256':sha256_file(FIXTURE)},
        'no_retries':True,'no_optimization':True}

def load_plan(path):
    plan=c0.strict_json(Path(path).read_text())
    if plan!=expected_plan(plan.get('kind')):raise ValueError('unreviewed calibration plan')
    exact=manifest(plan)
    if c0.strict_json(Path(plan['manifest']).read_text())!=exact:raise ValueError('calibration source/manifest mismatch')
    if Path(plan['output']).exists():raise ValueError('fresh output directory already exists')
    return plan,exact

def runtime_files(arm):
    plan=base_plan(arm)
    return c0.strict_json(Path(plan['runtime_baseline']).read_text()) if arm=='U' else c0.strict_json(Path(plan['runtime_receipt']).read_text())['runtime']

def preflight(plan,exact,launch,swap):
    if manifest(plan)!=exact:raise ValueError('source/contract changed before launch')
    conf=Config(**launch['configuration']);arm=launch['arm'];baseline=base_plan(arm)
    if plan['kind']=='equivalence':
        approved=c0.strict_json(HARNESS_RECEIPT.read_text())[arm]
        actual=build_dependencies(harness_path(arm),conf)
        atomic_json(Path(conf.output)/'harness-identity.json',actual)
        if actual['executable']!=approved['executable'] or actual['dependencies']!=approved['dependencies']:
            raise ValueError('correctness harness/dependency mismatch')
    # Each arm has its own exact runtime checker; never suppress a mismatch.
    return c1.preflight(conf,baseline,'E',swap) if arm=='U' else timing.preflight(conf,baseline,swap)

def samplers(conf,pid,arm):
    funcs=sampler_factory(conf,pid)
    funcs['maps']=partial(timing.runtime_maps,pid,conf,{f['path'] for f in runtime_files(arm)})
    return funcs

def process_timing(directory):
    records=c1.read_records(Path(directory)/'events.jsonl')
    release=[r['monotonic'] for r in records if r.get('event')=='exec_release']
    reap=[r['monotonic'] for r in records if r.get('event')=='reaped']
    if len(release)!=1 or len(reap)!=1 or not release[0]<reap[0]:
        raise ValueError('missing/invalid independent process interval')
    return {'exec_release_to_reaped_seconds':reap[0]-release[0],
        'isolated_model_loading_seconds':None,'isolated_trace_flush_seconds':None,
        'note':'Child wall time includes loading, warmup, workloads and exit/serialization; do not label residual time as loading.'}

def validate(conf,arm,kind,stdout,stderr):
    out=Path(conf.output);errors=[];result={'errors':errors}
    try:
        if kind=='overhead':
            rows=c0.validate_rows(stdout,conf,POLICY,reps=3)
            result['samples_ns']={('pp512' if r['n_prompt']==512 else 'tg128'):r['samples_ns'] for r in rows}
        else:
            exit_record=c0.strict_json(stdout)
            if exit_record!={'schema':'heterollm.equivalence.exit.v1','complete':True,'rows':129} or exit_record.get('complete') is not True or type(exit_record.get('rows')) is not int or 'EQUIVALENCE_ERROR' in stderr:
                raise ValueError('missing correctness exit record/error')
            result['capture']=analysis.validate_capture(out,arm)
        if arm=='T':
            for name,cap in [('cpu-trace.jsonl',64*1024**2),('device-timing.jsonl',32*1024**2)]:
                p=out/name
                if p.is_symlink() or p.stat().st_size>cap:raise ValueError('invalid trace path/size')
            cpu=(out/'cpu-trace.jsonl').read_text();header=c0.strict_json(cpu.splitlines()[0])
            for key,value in {'schema':'heterollm.cpu_trace.v2','capacity':65536,'record_bytes':176,'aggregate_capacity':32,'aggregate_record_bytes':592}.items():
                if header.get(key)!=value:raise ValueError('wrong CPU recorder build contract')
            from .trace_analysis import analyze
            trace=analyze(cpu,stderr);atomic_json(out/'trace-summary.json',trace)
            timed=analyze_timing((out/'device-timing.jsonl').read_text(),cpu,stderr,repetitions=3 if kind=='overhead' else 1)
            atomic_json(out/'timing-summary.json',timed);result['timing_complete']=True
        elif any((out/name).exists() for name in ('cpu-trace.jsonl','device-timing.jsonl')) or 'HETEROLLM_' in stderr:
            raise ValueError('unexpected instrumentation in U')
    except (OSError,ValueError,TypeError,KeyError,UnicodeError) as e:errors.append('workload/capture: '+str(e))
    try:
        pre=c0.strict_json((out/'preflight.json').read_text())
        result['allocation']=c1.allocation_check(out,POLICY,pre['c0_reference']['profile'])
        allowed={f['path'] for f in runtime_files(arm)};loaded=set()
        for record in c1.read_records(out/'maps.jsonl'):
            if record.get('error') or not record.get('data'):raise ValueError('invalid library sample')
            paths=set(record['data'].get('loaded_runtime_libraries',[]))
            if not paths<=allowed:raise ValueError('unapproved runtime mapping')
            loaded.update(paths)
        if not all(str((Path(conf.bench).parent/name).resolve()) in loaded for name in timing.REQUIRED_LIBS):
            raise ValueError('required loaded-library evidence missing')
        result['loaded_runtime_libraries']=sorted(loaded)
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as e:errors.append('placement/libraries: '+str(e))
    return result

def execute(plan,exact):
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+TOTALS[plan['kind']];signals=[]
    old={s:signal.signal(s,lambda sig,frame:signals.append(sig)) for s in (signal.SIGTERM,signal.SIGINT)}
    result={'status':'incomplete','kind':plan['kind'],'launches':0,'attempts':[],'started':stamp()};observations=[]
    def check():
        if signals:raise InterruptedError('suite_signal')
        if time.monotonic()>=deadline:raise TimeoutError('suite_deadline')
    try:
        atomic_json(out/'manifest.json',exact);atomic_json(out/'plan.json',plan);atomic_json(out/'suite-result.json',result)
        mem=parse_meminfo(Path('/proc/meminfo').read_text());swap=mem['SwapTotal']-mem['SwapFree']
        atomic_json(out/'suite-baseline.json',{'memory':mem,'swap_used':swap,**stamp()})
        for launch in exact['launches']:
            check();conf=Config(**launch['configuration'])
            ready=c1.cooldown(replace(conf,output=str(out/f"cooldown-{launch['ordinal']:02d}")),deadline,swap,check)
            append_json(out/'suite-events.jsonl',{'event':'cooldown','ordinal':launch['ordinal'],'result':ready,**stamp()})
            if ready['status']!='ready' or not ready['reader_cleanup_complete']:raise RuntimeError('cooldown failed')
            check()
            if deadline-time.monotonic()<LIMITS.overall:raise TimeoutError('insufficient full attempt budget')
            run=supervise(launch['argv'],conf.output,launch['environment'],partial(preflight,plan,exact,launch,swap),
                partial(samplers,conf,arm=launch['arm']),partial(validate,conf,launch['arm'],plan['kind']),LIMITS,
                scope='one independent calibration process; no retries')
            result['launches']+=run['launches'];result['attempts'].append({'launch':launch,'result':run})
            atomic_json(out/'suite-result.json',result)
            resources=c0.resource_summary(conf.output)
            resources['scope']='Calibration resources; sampled peaks except wait4 lifetime RSS. Numerical capture timings are not benchmarks.'
            atomic_json(Path(conf.output)/'resource-summary.json',resources)
            if run['status']!='passed' or not run['cleanup_complete'] or not run['reader_cleanup_complete']:
                raise RuntimeError('calibration process failed')
            elapsed=run['elapsed_seconds']
            if type(elapsed)not in (int,float) or not math.isfinite(elapsed) or not 0<=elapsed<=LIMITS.overall:
                raise TimeoutError('attempt deadline including validation/cleanup')
            wall=process_timing(conf.output);atomic_json(Path(conf.output)/'process-timing.json',wall)
            if plan['kind']=='overhead':
                record={'block':launch['block'],'arm':launch['arm'],'run_id':conf.output,'status':'passed','contract_sha256':digest(exact),
                    'samples_ns':run['validation']['samples_ns'],'process_timing':wall,'resources':resources}
                observations.append(record);atomic_json(out/'observations.json',observations)
        check()
        if plan['kind']=='overhead':summary=analysis.paired_summary(observations)
        else:
            summary=analysis.compare_captures(exact['launches'][0]['output'],exact['launches'][1]['output'])
        atomic_json(out/'comparison.json',summary)
        if plan['kind']=='equivalence' and not summary['passed']:raise ValueError('numerical equivalence failed; no tolerance adjustment')
        check();result['status']='passed'
    except BaseException as e:result.update(status='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
    finally:
        for s,h in old.items():signal.signal(s,h)
        result.update(ended=stamp(),elapsed_seconds=time.monotonic()-start,received_signals=signals)
        atomic_json(out/'suite-result.json',result)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv);plan,exact=load_plan(args.plan)
    if not args.execute:
        print(json.dumps({'mode':'preview_only','manifest':exact},indent=2));return 0
    with execution_lock():result=execute(plan,exact)
    print(json.dumps(result,indent=2));return 0 if result['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
