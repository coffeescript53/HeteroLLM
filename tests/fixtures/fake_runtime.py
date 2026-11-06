#!/usr/bin/env python3
"""CPU-only protocol stand-in. Never real inference or performance evidence."""
import bisect
import itertools
import json
import sys

args=sys.argv[1:]
def val(flag,default=None):
    return args[args.index(flag)+1] if flag in args else default
is_bench='-r' in args
sep='/' if is_bench else ','
devices=val('-dev',val('--device')).split(sep)
ratios=[float(v) for v in val('-ts',val('--tensor-split',sep.join(['1']*len(devices)))).split(sep)]
ngl=int(val('-ngl'))
start=max(3-ngl,0)
counts=min(ngl,3)
boundaries=list(itertools.accumulate(x/sum(ratios) for x in ratios))
for i in range(3):
    d='CPU' if i<start else devices[min(bisect.bisect_right(boundaries,(i-start)/counts),len(devices)-1)]
    print(f'load_tensors: layer {i:3d} assigned to device {d}, is_swa = 0',file=sys.stderr)
for d in devices:
    print(f'load_tensors: {d} model buffer size = 10.00 MiB',file=sys.stderr)
    print(f'node #  1 (   MUL_MAT): dense (  1M) [{d} weights] use=1,c=1:',file=sys.stderr)
print('llama_context: n_ctx = 512',file=sys.stderr)
print('llama_kv_cache: CPU KV buffer size = 1.0 MiB',file=sys.stderr)
if not is_bench:
    print('42')
else:
    reps=int(val('-r'))
    base={'n_batch':int(val('-b')),'n_ubatch':int(val('-ub')),'n_threads':int(val('-t')),
          'type_k':val('-ctk'),'type_v':val('-ctv'),'flash_attn':0,'fit_target':0,'fit_min_ctx':0,
          'split_mode':'layer','load_mode':'mmap','lazy_mode':'off','no_kv_offload':False,
          'no_op_offload':0,'no_host':False,'repack':True,'embeddings':False,'n_cpu_moe':0,
          'tensor_buft_overrides':'none','n_depth':int(val('-d')),'model_filename':val('-m'),
          'devices':'/'.join(devices),'n_gpu_layers':ngl,'tensor_split':val('-ts','0.00')}
    # CUDA-only is deliberately fastest to exercise unbiased selection.
    duration=1_000_000_000 if devices==['CUDA0'] else 2_000_000_000
    rows=[{**base,'n_prompt':p,'n_gen':n,'samples_ns':[duration+i*1000 for i in range(reps)]}
          for p,n in ((int(val('-p')),0),(0,int(val('-n'))))]
    print(json.dumps(rows))
