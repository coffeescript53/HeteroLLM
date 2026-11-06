#!/usr/bin/env python3
"""Create/verify a source-bound review proposal. No model execution API exists."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from heterollm import host_thermal, stage_c, stage_c1
from heterollm.config import Config

PLAN = ROOT/'configs/stage-c2-trace-proposal.json'
MANIFEST = ROOT/'configs/stage-c2-trace-proposal-manifest.json'
OUTPUT = ROOT/'runs/llama33-70b/stage-c2-trace-diagnostic-20261025'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proposal():
    config = Config.load(ROOT/'configs/stage-c1-memtotal-fixed.json')
    argv = stage_c1.command(config,stage_c.POLICIES[0])
    argv[0] = str(ROOT/'runtime-c2-trace/build-mixed/bin/llama-bench')
    argv[argv.index('-r')+1] = '1'
    env = stage_c1.environment(config)
    env['HETEROLLM_TRACE_FILE'] = str(OUTPUT/'cpu-trace.jsonl')
    return {'stage':'C2_TRACE_PROPOSAL_ONLY', 'execution_enabled':False,
            'approval':'NOT AUTHORIZED; requires independent review and explicit owner approval',
            'maximum_model_processes':1, 'output':str(OUTPUT),
            'suite_deadline_seconds':1500, 'argv':argv, 'environment':env,
            'limits':asdict(stage_c.C0_LIMITS),
            'cooldown':{'minimum_seconds':30,'maximum_seconds':600,'stable_seconds':10,
                        'fresh_readings':11,'both_gpus_max_c':45},
            'host_thermal':{'policy':host_thermal.POLICY,'sustained_seconds':25,
                            'dimm_admission_cooldown_seconds':120,'dimm_safe_readings':2,
                            'critical_alarm_or_missing_sensor':'immediate stop'},
            'gpu_reserve_mib':config.gpu_reserve_mib,'host_reserve_mib':config.host_reserve_mib,
            'model':{'path':config.model,'sha256':config.model_sha256,'bytes':config.model_size_bytes},
            'base_revision':config.source_revision,'physical_devices':config.devices,
            'candidate_binary_sha256':None, 'candidate_dependencies_sha256':None,
            'trace_scope':'aggregate CPU path evidence; phase and GPU completion unknown',
            'blocking_gates':['Build candidate in isolated checkout and bind complete runtime/dependency hashes',
                              'Implement/review one-process adapter retaining current preflight and supervisor',
                              'Require complete trace plus valid PP512/TG128 r1 and allocation/cleanup validation',
                              'Review final executable manifest and obtain explicit GPU authorization'],
            'phase_gpu_extensions':'Detailed design only; not required for aggregate routing, required before phase/timeline optimization claims'}


def manifest(plan):
    files = [p for folder in ('heterollm','tests','instrumentation') for p in (ROOT/folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    files += list((ROOT/'docs').glob('C2_*.md'))
    files += [ROOT/'scripts/analyze-c2-trace.py',Path(__file__),
              ROOT/'configs/stage-c0-hardware-baseline.json',ROOT/'configs/stage-c1-reference.json',
              ROOT/'configs/stage-c1-memtotal-fixed.json',ROOT/'configs/stage-c1-memtotal-fixed-manifest.json']
    original = json.loads((ROOT/'configs/stage-c1-memtotal-fixed-plan.json').read_text())
    files += [Path(original[k]) for k in ('receipt','runtime_baseline','dependency_baseline')]
    runtime = ROOT/'llama.cpp'
    files += [runtime/p for p in ('ggml/src/ggml-backend.cpp','ggml/src/ggml-cuda/ggml-cuda.cu',
               'ggml/src/ggml-cuda/vendors/hip.h','ggml/src/ggml-hip/CMakeLists.txt',
               'src/llama-context.cpp','src/llama-model.cpp','tools/llama-bench/llama-bench.cpp',
               'build-mixed/CMakeCache.txt','build-mixed/build.ninja')]
    return {'kind':'source-bound proposal, NOT executable runtime authorization',
            'plan_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write',action='store_true',help='write local proposal/manifest only; never execute')
    a=p.parse_args()
    value=proposal();bound=manifest(value)
    if a.write:
        PLAN.write_text(json.dumps(value,indent=2)+'\n')
        MANIFEST.write_text(json.dumps(bound,indent=2)+'\n')
        (ROOT/'docs/C2_PROPOSED_COMMAND.txt').write_text(
            'PROPOSAL ONLY. Raw child argv; do not run directly. No reviewed C2 launch adapter/build yet.\n'
            +shlex.join(value['argv'])+'\n\nExact child environment:\n'+json.dumps(value['environment'],indent=2)+'\n')
    if json.loads(PLAN.read_text()) != value or json.loads(MANIFEST.read_text()) != bound:
        raise ValueError('source/proposal mismatch; review and explicitly regenerate')
    if OUTPUT.exists(): raise ValueError('proposed output already exists')
    print(json.dumps({'source_bound_proposal_verified':True,'files':len(bound['files']),
                      'output_absent':True,'execution_enabled':False,
                      'gpu_processes_started':0,'manifest_sha256':sha(MANIFEST)},indent=2))


if __name__=='__main__':main()
