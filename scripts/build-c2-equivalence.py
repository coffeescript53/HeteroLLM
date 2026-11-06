#!/usr/bin/env python3
"""Compile two small CPU host harnesses; never execute a model program."""
import importlib.util
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/llama33-70b/c2-calibration-equivalence-prep-20261027'
sys.path.insert(0,str(ROOT))
from heterollm.supervisor import atomic_json

def commands():
    return [[ '/usr/bin/c++','-std=c++17','-O2','-Wall','-Wextra',
        '-I'+str(ROOT/'llama.cpp/include'),'-I'+str(ROOT/'llama.cpp/ggml/include'),
        str(ROOT/'instrumentation/calibration/equivalence.cpp'),
        '-L'+str(ROOT/runtime/'build-mixed/bin'),'-Wl,-rpath,'+str(ROOT/runtime/'build-mixed/bin'),
        '-lllama','-lggml','-lggml-base','-ldl','-o',str(OUT/'bin'/arm/'equivalence-capture')]
        for arm,runtime in [('U','llama.cpp'),('T','runtime-c2-phase-v3')]]

def compile_only():
    for command in commands():
        target=Path(command[-1]);target.parent.mkdir(parents=True,exist_ok=False)
        print('COMPILER ARGV:',repr(command),flush=True)
        subprocess.run(command,check=True,timeout=60)

def main():
    if sys.argv[1:]==['--compile-child']:
        compile_only();return 0
    if sys.argv[1:]:raise ValueError('no options supported')
    atomic_json(OUT/'HARNESS_BUILD_COMMANDS.json',{'commands':commands(),'maximum_parallel_jobs':1,'compiler_timeout_seconds_each':60,'GPU_execution':False})
    spec=importlib.util.spec_from_file_location('monitored_build',ROOT/'scripts/build-c2-runtime.py')
    build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
    build.OUT=OUT
    build.configure_command=lambda:[sys.executable,str(Path(__file__).resolve()),'--compile-child']
    sys.argv=[sys.argv[0],'configure']
    return build.main()

if __name__=='__main__':raise SystemExit(main())
