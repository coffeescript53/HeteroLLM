"""One source-bound C2 routing diagnostic. Preview unless explicitly executed."""
import argparse
from dataclasses import asdict, replace
from functools import partial
import json
import math
import os
from pathlib import Path
import re
import signal
import time
import traceback

from . import host_thermal, stage_c as c0, stage_c1 as c1
from .config import Config, digest
from .hardware import clean_environment, discover, file_identity, parse_meminfo, query, runtime_identity, sha256_file
from .pilot import maps_sample, sampler_factory
from .runner import execution_lock
from .runtime import require_capabilities
from .supervisor import atomic_json, append_json, stamp, supervise
from .trace_analysis import analyze

ROOT=c0.ROOT
SOURCE=ROOT/'runtime-c2-trace-v2'
BIN=SOURCE/'build-mixed/bin'
OUTPUT=ROOT/'runs/llama33-70b/stage-c2-routing-v2-diagnostic-20261026'
RECEIPT=ROOT/'configs/stage-c2-v2-runtime-receipt.json'
PATCH=ROOT/'instrumentation/v2/ggml-cpu-trace.patch'
PATCH_SHA256='891daaada23a5e95aa6cf749e30c93f93b2726b49fb9f2b718e16c0dce8b0fe4'
POLICY=c0.POLICIES[0]
LIMITS=c0.C0_LIMITS
SUITE_SECONDS=1500
TRACE_CAP=64*1024**2
REQUIRED_LIBS=('libggml-base.so','libggml-cuda.so','libggml-hip.so','libllama.so')


def expected_config():
    base=Config.load(ROOT/'configs/stage-c1-memtotal-fixed.json')
    return replace(base,source=str(SOURCE),bench=str(BIN/'llama-bench'),cli=str(BIN/'llama-cli'),
                   output=str(OUTPUT),max_launches=1,suite_seconds=SUITE_SECONDS,
                   environment={'LD_LIBRARY_PATH':str(BIN)+':/opt/rocm/core-10.0/lib'})


def command(config):
    argv=c0.command(config,POLICY);argv.remove('-v')
    return argv


def environment(config):
    return {**c1.environment(config),'HETEROLLM_TRACE_FILE':str(Path(config.output)/'cpu-trace.jsonl')}


def source_identity():
    def git(*args):
        r=query(['git','-C',str(SOURCE),*args],timeout=10)
        if r['exit_code']!=0:raise ValueError('candidate git identity unavailable')
        return r['stdout']
    revision=git('rev-parse','HEAD').strip()
    if revision!='5e03bdd8700948b9c41c54dd1b00f28a2aebc03f' or sha256_file(PATCH)!=PATCH_SHA256:
        raise ValueError('candidate base/patch identity mismatch')
    tracked=git('ls-files','-z').strip('\x00').split('\x00')
    extra=git('ls-files','--others','--exclude-standard','-z').strip('\x00').split('\x00')
    if extra!=['ggml/src/heterollm-cpu-trace.h']:
        raise ValueError('unexpected candidate untracked source')
    return {'revision':revision,'patch_sha256':PATCH_SHA256,'status':git('status','--porcelain'),
            'diff_sha256':digest(git('diff','--binary')),
            'files':{name:sha256_file(SOURCE/name) for name in sorted(tracked+extra)}}


def candidate_dependencies(config):
    files={};resolution={}
    paths={Path(config.cli).resolve(),Path(config.bench).resolve(),
           *[p.resolve() for p in BIN.glob('*.so*') if p.is_file()]}
    for path in sorted(paths):
        r=query(['ldd',str(path)],clean_environment(config.environment),timeout=5)
        resolution[str(path)]=r
        if r['exit_code']!=0 or 'not found' in r['stdout']:raise ValueError('unresolved candidate dependency')
        for name in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(',r['stdout'],re.M):
            p=Path(name).resolve()
            if ROOT/'llama.cpp' in p.parents:raise ValueError('candidate resolved original runtime library')
            if p.name.startswith(('libggml','libllama')) and p.parent!=BIN:
                raise ValueError('candidate library escapes isolated build')
            if str(p) not in files:files[str(p)]=file_identity(p)
    return {'resolution':resolution,'files':files}


def build_receipt(config):
    """Read-only identity capture after monitored compilation; no model calls."""
    for name in REQUIRED_LIBS:
        if not (BIN/name).is_file() or (BIN/name).resolve().parent!=BIN:raise ValueError('candidate library missing')
    runtime=runtime_identity(config)
    if any(Path(f['path']).parent!=BIN for f in runtime):raise ValueError('candidate runtime path escape')
    ninja=(SOURCE/'build-mixed/build.ninja').read_text()
    if '-DHETEROLLM_CPU_TRACE=1' not in ninja:raise ValueError('trace compilation definition missing')
    return {'source':source_identity(),'runtime':runtime,'dependencies':candidate_dependencies(config),
            'build_files':{str(p):sha256_file(p) for p in
                (SOURCE/'build-mixed/CMakeCache.txt',SOURCE/'build-mixed/build.ninja')},
            'trace_compile_definition':'HETEROLLM_CPU_TRACE=1'}


def verify_candidate(config, plan, save):
    approved=c0.strict_json(Path(plan['runtime_receipt']).read_text())
    actual=save('candidate-identity.json',build_receipt(config))
    if actual['source']!=approved['source']:raise ValueError('candidate source identity mismatch')
    if actual['runtime']!=approved['runtime']:raise ValueError('candidate runtime identity mismatch')
    if actual['dependencies']['files']!=approved['dependencies']['files']:
        raise ValueError('candidate dependency identity mismatch')
    if actual['build_files']!=approved['build_files']:raise ValueError('candidate build configuration mismatch')
    snapshot=save('hardware.json',discover(config))  # help, version, device list only
    require_capabilities(snapshot)
    if (snapshot['source']['exit_code']!=0 or snapshot['source']['stdout'].strip()!=config.source_revision
        or snapshot['source_status']['exit_code']!=0
        or snapshot['source_status']['stdout']!=approved['source']['status']):
        raise ValueError('candidate source revision/status mismatch')
    for name,record in snapshot['versions'].items():
        commit=re.search(r'commit ([0-9a-f]{7,40})',record['stdout']+record['stderr'])
        if record['exit_code']!=0 or not commit or not config.source_revision.startswith(commit[1]):
            raise ValueError(name+': candidate version mismatch')
    return snapshot


def manifest(config,plan):
    bound={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'heterollm').glob('*.py'))}
    for name in ('config','receipt','runtime_receipt','hardware_baseline','reference'):
        p=Path(plan[name]);bound[str(p.relative_to(ROOT))]=sha256_file(p)
    for p in (PATCH,ROOT/'configs/stage-c1-memtotal-fixed.json'):
        bound[str(p.relative_to(ROOT))]=sha256_file(p)
    child=replace(config,output=str(Path(config.output)/'01-equal-routing'))
    return {'stage':'C2_ROUTING','maximum_launches':1,'argv':command(child),
            'environment':environment(child),'limits':asdict(LIMITS),'suite_seconds':SUITE_SECONDS,
            'source_and_receipt_sha256':bound,'plan_digest':digest({k:v for k,v in plan.items() if k!='manifest'}),
            'cooldown':{'minimum_seconds':30,'maximum_seconds':600,'stable_seconds':10,'fresh_samples':11,'max_c':45},
            'host_thermal':{'policy':host_thermal.POLICY,'sustained_seconds':25,'admission_max_seconds':120},
            'trace':{'schema':'heterollm.cpu_trace.v2','cap_bytes':TRACE_CAP,'zero_fallback_valid':True},
            'interpretation':'Routing and CPU call evidence only; GPU time, overlap and phase unknown',
            'output':config.output,'child_output':child.output}


def load_plan(path):
    plan=c0.strict_json(Path(path).read_text())
    required={'stage','config','receipt','runtime_receipt','hardware_baseline','reference',
              'reference_sha256','mem_total_bytes','manifest','authorization'}
    if set(plan)!=required or plan['stage']!='C2_ROUTING' or plan['mem_total_bytes']!=98774167552:
        raise ValueError('only the single C2 routing plan is supported')
    config=Config.load(plan['config'])
    if config!=expected_config() or Path(plan['runtime_receipt'])!=RECEIPT:
        raise ValueError('unreviewed C2 configuration/runtime receipt')
    if c0.strict_json(Path(plan['manifest']).read_text())!=manifest(config,plan):
        raise ValueError('C2 source/manifest/receipt mismatch')
    if sha256_file(plan['reference'])!=plan['reference_sha256']:raise ValueError('C0 reference changed')
    return config,plan


def preflight(config,plan,swap):
    # Reuse all original model, hardware, MemTotal, RAM, capacity, swap and sensor checks.
    reference=c1.verify_reference(plan)
    result=c0.preflight(config,plan,POLICY,swap,runtime_check=verify_candidate)
    predicted={str(i):d for d,ids in result['screening']['estimated_layers'].items() for i in ids}
    predicted.update({str(i):'CPU' for i in range(81) if str(i) not in predicted})
    if predicted!=reference['profiles']['E']['layers']:raise ValueError('E placement changed from C0')
    result['c0_reference']={'profile':reference['profiles']['E'],'id':'E','fresh_per_layer_trace':False}
    return result


def runtime_maps(pid,config,allowed):
    value=maps_sample(pid,config.model)
    try:raw=(Path('/proc')/str(pid)/'maps').read_text()
    except (FileNotFoundError,ProcessLookupError):
        if value.get('state')=='zombie':raw=''
        else:raise
    paths=sorted({line.split(maxsplit=5)[5] for line in raw.splitlines()
                  if len(line.split(maxsplit=5))==6 and
                  any(token in line.split(maxsplit=5)[5] for token in ('/libggml','/libllama'))})
    if any(p not in allowed for p in paths):raise ValueError('unapproved loaded runtime library')
    value['loaded_runtime_libraries']=paths
    return value


def samplers(config,pid,plan):
    funcs=sampler_factory(config,pid)
    receipt=c0.strict_json(Path(plan['runtime_receipt']).read_text())
    allowed={f['path'] for f in receipt['runtime']}
    funcs['maps']=partial(runtime_maps,pid,config,allowed)
    return funcs


def validate(config,stdout,stderr):
    errors=[];workloads=[];trace=None;allocation=None;loaded=[]
    directory=Path(config.output)
    try:workloads=c0.validate_rows(stdout,config,POLICY,reps=1)
    except (ValueError,TypeError,KeyError,AttributeError) as e:errors.append('workload: '+str(e))
    try:
        path=directory/'cpu-trace.jsonl'
        if path.is_symlink() or path.stat().st_size>TRACE_CAP:raise ValueError('trace path/size invalid')
        trace=analyze(path.read_text(),stderr)
        records=[c0.strict_json(line) for line in path.read_text().splitlines()]
        if (records[0]['schema']!='heterollm.cpu_trace.v2' or records[0]['capacity']!=65536
            or records[0]['record_bytes']!=176 or records[0]['aggregate_capacity']!=32
            or records[0]['aggregate_record_bytes']!=592):
            raise ValueError('trace recorder build contract differs')
        submitted={r['src'] for r in records if r.get('kind')=='graph.submit'}
        if not {'CUDA0','ROCm0'}<=submitted or not trace['kinds'].get('graph.compute_splits'):
            raise ValueError('trace lacks two-backend scheduler submission evidence')
        atomic_json(directory/'trace-summary.json',trace)
    except (OSError,ValueError,TypeError,KeyError,UnicodeError) as e:errors.append('trace: '+str(e))
    try:
        pre=c0.strict_json((directory/'preflight.json').read_text())
        allocation=c1.allocation_check(directory,POLICY,pre['c0_reference']['profile'])
        for record in c1.read_records(directory/'maps.jsonl'):
            if record.get('error') or not record.get('data'):raise ValueError('invalid runtime-map sample')
            loaded.extend(record['data'].get('loaded_runtime_libraries',[]))
        if not all(str((BIN/name).resolve()) in loaded for name in REQUIRED_LIBS):
            raise ValueError('candidate shared-library mapping evidence incomplete')
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as e:errors.append('allocation/libraries: '+str(e))
    return {'errors':errors,'diagnostic_workloads':workloads,'trace':trace,
            'placement_verification':allocation,'loaded_runtime_libraries':sorted(set(loaded)),
            'timings_eligible_for_comparison':False,'semantic_answer_test':False,
            'gpu_execution_time':None,'gpu_overlap':None,'phase_attribution':None}


def execute(config,plan):
    directory=Path(config.output);directory.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+SUITE_SECONDS;signals=[]
    old={s:signal.signal(s,lambda sig,frame:signals.append(sig)) for s in (signal.SIGTERM,signal.SIGINT)}
    result={'stage':'C2_ROUTING','status':'incomplete','launches':0,'started':stamp(),
            'timings_eligible_for_comparison':False}
    def check():
        if signals:raise InterruptedError('suite_signal')
        if time.monotonic()>=deadline:raise TimeoutError('suite_deadline')
    try:
        exact=manifest(config,plan);atomic_json(directory/'manifest.json',exact)
        atomic_json(directory/'plan.json',plan);atomic_json(directory/'suite-result.json',result)
        mem=parse_meminfo(Path('/proc/meminfo').read_text());swap=mem['SwapTotal']-mem['SwapFree']
        atomic_json(directory/'suite-baseline.json',{'memory':mem,'swap_used':swap,**stamp()})
        ready=c1.cooldown(replace(config,output=str(directory/'cooldown')),deadline,swap,check)
        result['cooldown']=ready;atomic_json(directory/'suite-result.json',result)
        if ready['status']!='ready' or not ready['reader_cleanup_complete']:raise RuntimeError('cooldown failed')
        check()
        if deadline-time.monotonic()<LIMITS.overall:raise TimeoutError('insufficient full attempt budget')
        child=replace(config,output=exact['child_output'])
        run=supervise(exact['argv'],child.output,exact['environment'],partial(preflight,child,plan,swap),
                      partial(samplers,child,plan=plan),partial(validate,child),LIMITS,
                      scope='one C2 CPU-routing trace diagnostic; no retry or follow-up')
        result.update(attempt=run,launches=run['launches'])
        atomic_json(directory/'suite-result.json',result)
        resources=c0.resource_summary(child.output)
        resources['scope']='C2 routing diagnostic; sampled peaks except wait4 RSS; no performance comparison'
        atomic_json(Path(child.output)/'resource-summary.json',resources)
        trace_path=Path(child.output)/'cpu-trace.jsonl'
        if trace_path.is_file():
            atomic_json(Path(child.output)/'trace-file-identity.json',file_identity(trace_path))
        if run['status']!='passed' or not run['cleanup_complete'] or not run['reader_cleanup_complete']:
            raise RuntimeError('routing diagnostic failed: '+str(run.get('reason')))
        elapsed=run['elapsed_seconds']
        if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<=elapsed<=LIMITS.overall:
            raise TimeoutError('attempt deadline including finalization exceeded or invalid')
        check();result['status']='passed'
    except BaseException as e:
        result.update(status='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
    finally:
        for s,h in old.items():signal.signal(s,h)
        result.update(elapsed_seconds=time.monotonic()-start,ended=stamp(),received_signals=signals)
        atomic_json(directory/'suite-result.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    parser.add_argument('--execute',action='store_true',help='Only after explicit approval of this one GPU diagnostic')
    args=parser.parse_args(argv);config,plan=load_plan(args.plan)
    if Path(config.output).exists():raise ValueError('fresh output directory already exists')
    if not args.execute:
        print(json.dumps({'mode':'preview_only','output_exists':False,'manifest':manifest(config,plan)},indent=2));return
    with execution_lock():result=execute(config,plan)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='passed' else 1)


if __name__=='__main__':main()
