from dataclasses import replace
import json
from pathlib import Path
import struct

from heterollm.config import Config

ROOT = Path(__file__).resolve().parents[1]


def config(**changes):
    return replace(Config.load(ROOT/'configs/llama33-70b.json'),**changes)


def snapshot():
    # Static fixture has no dependency on a live installation.
    flags = '--device <dev0/dev1/...> --split-mode --tensor-split --n-gpu-layers --fit-target --load-mode --lazy-mode --n-depth --cache-type-k --cache-type-v --offline'
    cli = '--device --split-mode --tensor-split --fit --single-turn --simple-io --offline'
    telemetry = {'memory':{'MemTotal':48*1024**3,'MemAvailable':40*1024**3,'SwapTotal':8*1024**3,'SwapFree':8*1024**3},
                 'nvidia':[{'pci':'0000:05:00.0','temperature_c':40,'driver':'fixture'}],
                 'amd':[{'pci':'0000:03:00.0','temperature_c':None}]}
    return {'synthetic':True,'devices':[{'name':'CUDA0','description':'NVIDIA GeForce RTX 5090','total_bytes':32*1024**3,'free_bytes':30*1024**3,'pci':'0000:05:00.0'},
                                       {'name':'ROCm0','description':'AMD Radeon AI PRO R9700','total_bytes':32*1024**3,'free_bytes':31*1024**3,'pci':'0000:03:00.0'}],
            'telemetry':telemetry,'bench_help':{'exit_code':0,'stdout':flags,'stderr':''},'cli_help':{'exit_code':0,'stdout':cli,'stderr':''},
            'versions':{name:{'exit_code':0,'stdout':'commit 5e03bdd87','stderr':''} for name in ('bench','cli')},
            'source':{'stdout':config().source_revision},'source_status':{'exit_code':0,'stdout':''},'platform':'CPU-only test fixture','cpu':{'stdout':'Model name: synthetic'}}


def benchmark_rows(c,reps=3):
    base = {'n_batch':c.batch,'n_ubatch':c.ubatch,'n_threads':c.threads,'type_k':'f16','type_v':'f16','flash_attn':0,
            'fit_target':0,'fit_min_ctx':0,'split_mode':'layer','load_mode':'mmap','lazy_mode':'off',
            'no_kv_offload':False,'no_op_offload':0,'no_host':False,'repack':True,'embeddings':False,
            'n_cpu_moe':0,'tensor_buft_overrides':'none','n_depth':c.depth}
    return [{**base,'n_prompt':p,'n_gen':n,'samples_ns':[1_000_000_000+i*10_000_000 for i in range(reps)]}
            for p,n in ((c.prompt_tokens,0),(0,c.generation_tokens))]


def tiny_gguf(path):
    def string(s):
        b=s.encode(); return struct.pack('<Q',len(b))+b
    meta={'general.architecture':'llama','llama.block_count':2,'llama.embedding_length':8,
          'llama.attention.head_count':2,'llama.attention.head_count_kv':1}
    b=b'GGUF'+struct.pack('<IQQ',3,4,len(meta))
    for k,v in meta.items():
        b+=string(k)+struct.pack('<I',8 if isinstance(v,str) else 4)
        b+=string(v) if isinstance(v,str) else struct.pack('<I',v)
    for i,n in enumerate(('token_embd.weight','blk.0.attn_q.weight','blk.1.attn_q.weight','output.weight')):
        b+=string(n)+struct.pack('<IQIQ',1,8,0,i*32)
    b+=b'\0'*((-len(b))%32)+b'\0'*128
    Path(path).write_bytes(b)


def thermal_fixture(cpu=40, dimms=(35,36)):
    """Synthetic records only; never reads host hwmon."""
    import time
    from heterollm.host_thermal import SENSOR_IDS
    sensors={}
    for key,value in zip(SENSOR_IDS,(cpu,*dimms)):
        kind='cpu' if key=='cpu' else 'dimm'
        sensors[key]={'kind':kind,'driver':'k10temp' if kind=='cpu' else 'spd5118',
                      'identity':'synthetic/'+key,'temperature_c':value,'alarms':{},'unit':'C'}
    return {'started_monotonic':time.monotonic(),'ended_monotonic':time.monotonic(),'sensors':sensors}


def c1_cpu_plan(directory):
    """Temporary current-source manifest for CPU contract tests, never authorization.

    Historical C1 plans are intentionally left bound to their original sources.
    """
    from heterollm import stage_c1 as c1
    from heterollm.config import Config
    directory=Path(directory)
    plan=json.loads((ROOT/'configs/stage-c1-memtotal-fixed-plan.json').read_text())
    data=json.loads(Path(plan['config']).read_text())
    data['output']=str(ROOT/'runs/llama33-70b'/('stage-c1-cpu-fixture-'+directory.name))
    (directory/'config.json').write_text(json.dumps(data))
    plan.update(config=str(directory/'config.json'),manifest=str(directory/'manifest.json'))
    (directory/'manifest.json').write_text(json.dumps(c1.manifest(Config.load(directory/'config.json'),plan)))
    path=directory/'plan.json';path.write_text(json.dumps(plan))
    return path
