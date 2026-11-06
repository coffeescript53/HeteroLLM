#!/usr/bin/env python3
"""Bounded, host-thermal-supervised build only; never invokes a model binary."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from heterollm.host_thermal import ThermalGuard, sample
from heterollm.hardware import parse_meminfo
from heterollm.supervisor import Reader, append_json, atomic_json, stamp, group_signal, live_group, death_signal

OUT=ROOT/'runs/llama33-70b/c2-build-adapter-prep-20261025'
SOURCE=ROOT/'runtime-c2-trace'
BUILD=SOURCE/'build-mixed'
GIB=1024**3


def configure_command():
    values={}
    for line in (ROOT/'llama.cpp/build-mixed/CMakeCache.txt').read_text().splitlines():
        if line.startswith(('#','//')) or '=' not in line or ':' not in line.split('=',1)[0]:continue
        key_type,value=line.split('=',1);key,kind=key_type.split(':',1)
        if kind in ('BOOL','STRING','UNINITIALIZED') and (key.startswith(('GGML_','LLAMA_')) or key in ('BUILD_SHARED_LIBS','BUILD_TESTING','GPU_TARGETS','CMAKE_BUILD_TYPE','CMAKE_CUDA_ARCHITECTURES')):
            values[key]=value
    values.update(CMAKE_C_COMPILER='/usr/bin/cc',CMAKE_CXX_COMPILER='/usr/bin/c++',
                  CMAKE_CUDA_COMPILER='/usr/local/cuda-13.2/bin/nvcc',
                  CMAKE_HIP_COMPILER='/opt/rocm/core-10.0/lib/llvm/bin/clang',
                  CMAKE_CXX_FLAGS='-DHETEROLLM_CPU_TRACE=1')
    return ['cmake','-S',str(SOURCE),'-B',str(BUILD),'-G','Ninja',
            *[f'-D{k}={v}' for k,v in sorted(values.items())]]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('configure','compile'))
    phase=p.parse_args().phase
    command=configure_command() if phase=='configure' else ['cmake','--build',str(BUILD),'--parallel','2','--target','llama-bench','llama-cli']
    directory=OUT/('build-'+phase);directory.mkdir(exist_ok=False)
    env={'PATH':'/usr/local/cuda-13.2/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin',
         'HOME':str(Path.home()),'LANG':'C.UTF-8','LC_ALL':'C.UTF-8',
         'LD_LIBRARY_PATH':'/opt/rocm/core-10.0/lib','ROCM_PATH':'/opt/rocm',
         'CMAKE_BUILD_PARALLEL_LEVEL':'2'}
    atomic_json(directory/'command.json',{'argv':command,'environment':env,'maximum_seconds':3600,'maximum_jobs':2})
    start=time.monotonic();child=None;reader=None;thermal=ThermalGuard();last=None
    result={'status':'incomplete','started':stamp(),'gpu_workload':False};signals=[]
    old={s:signal.signal(s,lambda sig,frame:signals.append(sig)) for s in (signal.SIGTERM,signal.SIGINT)}
    baseline=parse_meminfo(Path('/proc/meminfo').read_text());swap=baseline['SwapTotal']-baseline['SwapFree']
    previous_oom=None;next_mem=0
    try:
        atomic_json(directory/'result.json',result)
        reader=Reader('thermal',sample,directory,1,1)
        while True:
            now=time.monotonic()
            if signals:raise InterruptedError('build_signal')
            if now-start>=3600:raise TimeoutError('build_deadline')
            if child is None and now-start>=120:raise TimeoutError('build_admission_deadline')
            value=reader.poll()
            if value:
                last=value
                stop=thermal.observe(value['data'],time.monotonic(),admission=child is None)
                atomic_json(directory/'thermal-summary.json',thermal.summary())
                if stop:raise RuntimeError(stop)
            if last and now-last['started']['monotonic']>3:raise RuntimeError('build_thermal_stale')
            if now>=next_mem:
                mem=parse_meminfo(Path('/proc/meminfo').read_text());vm=dict(s.split() for s in Path('/proc/vmstat').read_text().splitlines());oom=int(vm['oom_kill'])
                rss=0
                if child is not None:
                    for pid in live_group(child.pid):
                        try:
                            status=(Path('/proc')/str(pid)/'status').read_text()
                        except (FileNotFoundError,ProcessLookupError):continue
                        rss+=next((int(s.split()[1])*1024 for s in status.splitlines() if s.startswith('VmRSS:')),0)
                append_json(directory/'memory.jsonl',{'memory':mem,'group_rss_bytes':rss,'oom_kill':oom,**stamp()})
                if mem['MemAvailable']<(40 if child is None else 4)*GIB:raise RuntimeError('build_ram_floor')
                if mem['SwapTotal']-mem['SwapFree']-swap>256*1024**2:raise RuntimeError('build_swap_growth')
                if mem.get('HardwareCorrupted',0) or (previous_oom is not None and oom>previous_oom):raise RuntimeError('build_memory_fault')
                if rss>32*GIB:raise RuntimeError('build_group_rss_limit')
                previous_oom=oom;next_mem=now+1
            if child is None and thermal.admission_safe_samples>=2:
                parent=os.getpid()
                with (directory/'stdout.txt').open('xb') as stdout,(directory/'stderr.txt').open('xb') as stderr:
                    child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,cwd=ROOT,
                                           start_new_session=True,preexec_fn=lambda:death_signal(parent))
                result['pid']=child.pid;append_json(directory/'events.jsonl',{'event':'build_started','pid':child.pid,**stamp()})
            if sum(p.stat().st_size for p in directory.glob('*.txt'))>256*1024**2:raise RuntimeError('build_log_limit')
            if child is not None and child.poll() is not None:
                result['exit_code']=child.returncode
                if child.returncode:raise RuntimeError('build_failed')
                result['status']='passed';break
            time.sleep(.02)
    except BaseException as e:
        result.update(status='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
        atomic_json(directory/'stop-intent.json',{**result,'thermal':thermal.summary(),**stamp()})
    finally:
        if child is not None:
            group_signal(child.pid,signal.SIGTERM)
            until=time.monotonic()+2
            while live_group(child.pid) and time.monotonic()<until:time.sleep(.02)
            group_signal(child.pid,signal.SIGKILL)
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
            result['cleanup_complete']=child.poll() is not None and not live_group(child.pid)
        else:result['cleanup_complete']=True
        result['reader_cleanup_complete']=reader.close() if reader else True
        if not result['cleanup_complete'] or not result['reader_cleanup_complete']:result['status']='failed'
        result.update(elapsed_seconds=time.monotonic()-start,ended=stamp(),thermal=thermal.summary(),signals=signals)
        atomic_json(directory/'result.json',result)
        for s,handler in old.items():signal.signal(s,handler)
    print(json.dumps(result,indent=2));return 0 if result['status']=='passed' else 1


if __name__=='__main__':sys.exit(main())
