"""One future phase/event diagnostic, reusing the reviewed supervisor adapter.

The private module instance isolates its exact runtime contract; the historical
v2 adapter and manifests are never modified or accepted as substitutes.
"""
import importlib.util
from pathlib import Path
from .hardware import sha256_file
from .config import digest
from .timing_analysis import analyze_timing, HEADER

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('heterollm._c2_timing_contract',ROOT/'heterollm/stage_c2.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
adapter.SOURCE=ROOT/'runtime-c2-phase-v3'
adapter.BIN=adapter.SOURCE/'build-mixed/bin'
adapter.OUTPUT=ROOT/'runs/llama33-70b/stage-c2-phase-timing-diagnostic-20261027'
adapter.RECEIPT=ROOT/'configs/stage-c2-timing-runtime-receipt.json'
adapter.PATCH=ROOT/'instrumentation/v3/phase-timing.patch'
adapter.PATCH_SHA256='e02995f6289bc8cbbcce6369bc5a7dcb97cee38c1ff1b0b4c043b46191dc4586'
_environment=adapter.environment
_manifest=adapter.manifest
_validate=adapter.validate

def source_identity():
    def git(*args):
        r=adapter.query(['git','-C',str(adapter.SOURCE),*args],timeout=10)
        if r['exit_code']!=0:raise ValueError('timing source identity unavailable')
        return r['stdout']
    revision=git('rev-parse','HEAD').strip()
    if revision!='5e03bdd8700948b9c41c54dd1b00f28a2aebc03f' or sha256_file(adapter.PATCH)!=adapter.PATCH_SHA256:raise ValueError('timing base/patch identity mismatch')
    tracked=git('ls-files','-z').strip('\0').split('\0');extra=git('ls-files','--others','--exclude-standard','-z').strip('\0').split('\0')
    expected=sorted('ggml/src/'+s for s in ('heterollm-cpu-trace.h','ht3-api.h','ht3-recorder.h','ht3-pool.h','ht3-cuda.h'))
    if sorted(extra)!=expected:raise ValueError('unexpected timing source files')
    return {'revision':revision,'patch_sha256':adapter.PATCH_SHA256,'status':git('status','--porcelain'),
            'diff_sha256':digest(git('diff','--binary')),'files':{p:sha256_file(adapter.SOURCE/p) for p in sorted(tracked+extra)}}

def environment(config):
    return {**_environment(config),'HETEROLLM_TIMING_FILE':str(Path(config.output)/'device-timing.jsonl')}

def manifest(config,plan):
    value=_manifest(config,plan)
    value['timing']={'header':HEADER,'cap_bytes':32*1024**2,'required_phases':['warmup_pp','pp512','warmup_tg','tg128'],'extra_waits':False}
    value['interpretation']='Completed same-device stream intervals and explicit phases; CPU enclosure uncertainty; kernel-only time, overlap, queue delay and bottleneck UNKNOWN'
    return value

def validate(config,stdout,stderr):
    value=_validate(config,stdout,stderr)
    value['completed_device_timing']=None
    try:
        path=Path(config.output)/'device-timing.jsonl'
        if path.is_symlink() or path.stat().st_size>32*1024**2:raise ValueError('timing path/size invalid')
        timing=analyze_timing(path.read_text(),(Path(config.output)/'cpu-trace.jsonl').read_text(),stderr)
        value['completed_device_timing']=timing
        value['phase_attribution']='explicit warmup_pp / pp512 / warmup_tg / tg128; event aggregates remain whole-process'
        adapter.atomic_json(Path(config.output)/'timing-summary.json',timing)
    except (OSError,ValueError,TypeError,KeyError,UnicodeError) as e:value['errors'].append('timing: '+str(e))
    return value

adapter.source_identity=source_identity
adapter.environment=environment
adapter.manifest=manifest
adapter.validate=validate

def main(argv=None):return adapter.main(argv)
if __name__=='__main__':main()
