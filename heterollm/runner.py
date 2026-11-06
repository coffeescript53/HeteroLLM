"""Bounded, sequential process execution and durable experiment evidence."""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from . import analysis
from .config import digest
from .gguf import read_layout
from .hardware import (MIB, clean_environment, discover, file_identity, runtime_identity,
                       query, telemetry)
from .planner import balanced_order, candidates, estimate
from .runtime import (PROMPT, bench_command, functional_command, placement, require_capabilities,
                      response_correct, validation_key, verify_placement)


class StopSuite(RuntimeError):
    """A resource/fault/deadline failure forbids automatic retries."""


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def unique_directory(root, label):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    name = f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}-{label}-{uuid.uuid4().hex[:8]}'
    path = root / name
    path.mkdir(exist_ok=False)
    return path


@contextmanager
def execution_lock():
    path = Path('/tmp') / f'heterollm-{os.getuid()}.lock'
    fd = os.open(path,os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,0o600)
    try:
        try:
            fcntl.flock(fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError('another HeteroLLM workload owns the execution lock') from e
        yield
    finally:
        os.close(fd)


def terminate_group(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid,sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    process.wait()


def classify(code, text, stop_reason=None):
    if stop_reason:
        return stop_reason
    lower = text.lower()
    if re.search(r'out of memory|hiperroroutofmemory|cuda.*alloc.*fail|failed to allocate|std::bad_alloc',lower):
        return 'memory_failure'
    if re.search(r'gpu fault|illegal memory access|device.*lost|hsa_status_error|hip error|cuda error',lower):
        return 'gpu_failure'
    if code == 0:
        return 'ok'
    if code is None:
        return 'launch_failure'
    return 'runtime_failure'


def run_process(argv, directory, timeout, environment, monitor=None, max_log_mib=256):
    """Own the process group; raw stdout/stderr survive failure or interruption."""
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    write_json(directory/'command.json',{'argv':argv,'environment_overrides':{k:v for k,v in environment.items() if k not in ('PATH','HOME','LANG','LC_ALL','TMPDIR')},'timeout_seconds':timeout})
    started = time.monotonic()
    proc = None
    reason, code, launch_error = None,None,None
    samples = 0
    previous_term = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    with (directory/'stdout.txt').open('x') as out, (directory/'stderr.txt').open('x') as err, (directory/'telemetry.jsonl').open('x') as observations:
        try:
            command = ['/usr/bin/time','-v','-o',str(directory/'resource-usage.txt'),*argv] if Path('/usr/bin/time').exists() else argv
            proc = subprocess.Popen(command,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                                    env=environment,start_new_session=True,cwd=directory)
            next_observation = 0.0
            while proc.poll() is None:
                elapsed = time.monotonic()-started
                if elapsed >= timeout:
                    reason = 'timeout'
                    break
                if out.tell()+err.tell() > max_log_mib*MIB:
                    reason = 'log_limit'
                    break
                if monitor is not None and elapsed >= next_observation:
                    sample, reason = monitor()
                    sample['elapsed_seconds'] = elapsed
                    json.dump(sample,observations,allow_nan=False)
                    observations.write('\n'); observations.flush()
                    samples += 1
                    next_observation = elapsed + 2
                    if reason:
                        break
                time.sleep(0.05)
            if reason:
                terminate_group(proc)
            code = proc.wait()
        except KeyboardInterrupt:
            reason = 'interrupted'
        except OSError as e:
            launch_error = str(e)
            reason = 'launch_failure'
        except Exception as e:
            launch_error = f'{type(e).__name__}: {e}'
            reason = 'monitor_failure'
        finally:
            if proc is not None:
                terminate_group(proc)
                code = proc.returncode
            signal.signal(signal.SIGTERM,previous_term)
    # Log cap bounds this read; preserve the full files separately.
    error = (directory/'stderr.txt').read_text(errors='replace')
    result = {'argv':argv,'exit_code':code,'classification':classify(code,error,reason),
              'elapsed_seconds':time.monotonic()-started,'telemetry_samples':samples,'launch_error':launch_error}
    write_json(directory/'process.json',result)
    return result


def safety_reason(sample, config, initial_swap):
    mem = sample['memory']
    if mem['MemAvailable'] < config.minimum_available_ram_mib*MIB:
        return 'host_memory_pressure'
    swap = mem['SwapTotal']-mem['SwapFree']
    if swap-initial_swap > config.max_swap_growth_mib*MIB:
        return 'swap_growth'
    selected = set(config.devices.values())
    for g in sample['nvidia'] + sample['amd']:
        if g['pci'] in selected and g.get('temperature_c') is not None and g['temperature_c'] >= config.max_temperature_c:
            return 'temperature_limit'
    return None


def verify_model(config, receipt):
    path = Path(config.model)
    if path.stat().st_size != config.model_size_bytes:
        raise ValueError('model size differs from pinned publisher metadata')
    identity = file_identity(path)
    if identity['sha256'] != config.model_sha256:
        raise ValueError('model SHA-256 differs from pinned publisher digest')
    layout = read_layout(path)
    write_json(receipt,{'identity':identity,'layout':layout.to_dict(),'verification':'full SHA-256 read'})
    return identity


def load_receipt(config, receipt):
    saved = json.loads(Path(receipt).read_text())
    current = file_identity(config.model,False)
    expected = saved['identity']
    if any(current[k] != expected[k] for k in ('path','size_bytes','mtime_ns')):
        raise ValueError('model changed since integrity verification; create a new receipt')
    if expected['sha256'] != config.model_sha256 or expected['size_bytes'] != config.model_size_bytes:
        raise ValueError('receipt does not match pinned model')
    return expected


class Session:
    def __init__(self, config, receipt):
        self.config = config
        self.started = time.monotonic()
        self.count = 0
        self.directory = unique_directory(config.output,'suite')
        self.records = []
        write_json(self.directory/'config.json',config.to_dict())
        self.model = load_receipt(config,receipt)
        self.layout = read_layout(config.model)
        self.snapshot = discover(config)
        write_json(self.directory/'hardware.json',self.snapshot)
        require_capabilities(self.snapshot)
        if self.snapshot['source']['stdout'].strip() != config.source_revision or self.snapshot['source_status']['exit_code'] != 0 or self.snapshot['source_status']['stdout'].strip():
            raise ValueError('runtime source revision is wrong or source tree is dirty')
        for name,version in self.snapshot['versions'].items():
            reported = re.search(r'commit ([0-9a-f]{7,40})',version['stdout']+version['stderr'])
            if version['exit_code'] != 0 or not reported or not config.source_revision.startswith(reported[1]):
                raise ValueError(f'{name} executable does not report the pinned source revision')
        self.runtime_files = runtime_identity(config)
        write_json(self.directory/'runtime-files.json',self.runtime_files)
        code_files = [file_identity(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
        write_json(self.directory/'orchestrator-files.json',code_files)
        # Dynamic backend dependencies are not necessarily linked by the launcher.
        dependencies = {str(p):query(['ldd',str(p)],clean_environment(config.environment))
                        for p in sorted(Path(config.bench).parent.glob('libggml-*.so'))}
        write_json(self.directory/'library-resolution.json',dependencies)
        if any('not found' in x['stdout'] for x in dependencies.values()):
            raise ValueError('an installed backend has unresolved shared-library dependencies')
        write_json(self.directory/'model.json',{'identity':self.model,'layout':self.layout.to_dict()})
        initial = self.snapshot['telemetry']['memory']
        self.initial_swap = initial['SwapTotal']-initial['SwapFree']
        self.comparison_key = digest({'model':self.model['sha256'],'runtime':self.runtime_files,
                                      'workload':config.workload(),'devices':config.devices,
                                      'host_total':initial['MemTotal'],'environment':config.environment,
                                      'platform':self.snapshot['platform'],
                                      'cpu':[line for line in self.snapshot['cpu']['stdout'].splitlines()
                                             if line.startswith(('Model name:','CPU(s):','Thread(s) per core:','Core(s) per socket:','Socket(s):'))],
                                      'nvidia_drivers':[x['driver'] for x in self.snapshot['telemetry']['nvidia']]})
        self.validations = {}

    def monitor(self):
        sample = telemetry()
        reason = safety_reason(sample,self.config,self.initial_swap)
        if time.monotonic()-self.started >= self.config.suite_seconds:
            reason = 'suite_deadline'
        return sample,reason

    def launch(self, argv, path, diagnostic=False):
        if self.count >= self.config.max_launches:
            raise StopSuite('maximum launch count reached')
        remaining = self.config.suite_seconds - (time.monotonic()-self.started)
        if remaining <= 0:
            raise StopSuite('suite wall-clock deadline reached')
        # Recheck model stat identity, resource availability and configured devices.
        if file_identity(self.config.model,False) | {'sha256':self.model['sha256']} != self.model:
            raise StopSuite('model identity changed during execution')
        before, reason = self.monitor()
        if reason:
            raise StopSuite(reason)
        self.count += 1
        print(f'[{self.count}/{self.config.max_launches}] {path.parent.name}/{path.name}', file=sys.stderr, flush=True)
        result = run_process(argv,path,min(self.config.timeout_seconds,remaining),
                             clean_environment(self.config.environment,diagnostic),self.monitor,self.config.max_log_mib)
        print(f'  {result["classification"]}; exit={result["exit_code"]}; elapsed={result["elapsed_seconds"]:.1f}s',
              file=sys.stderr, flush=True)
        if result['classification'] in {'memory_failure','gpu_failure','host_memory_pressure','swap_growth','temperature_limit',
                                        'monitor_failure','interrupted','suite_deadline','log_limit','timeout'}:
            raise StopSuite(f'{result["classification"]}: {path}')
        return result

    def validate(self, policy):
        folder = unique_directory(self.directory,policy.name+'-validation')
        snapshot = discover(self.config)
        screening = estimate(policy,self.layout,snapshot,self.config)
        write_json(folder/'preflight.json',{'snapshot':snapshot,'screening':screening.to_dict()})
        if not screening.admissible:
            raise ValueError(f'policy rejected: {screening.reasons}')
        key = validation_key(self.config,policy,self.model,self.runtime_files)
        write_json(folder/'prompt.json',{'prompt':PROMPT,'expected':'42','seed':42,'temperature':0,'max_tokens':8})
        functional = self.launch(functional_command(self.config,policy),folder/'functional',True)
        errors = []
        if functional['classification'] != 'ok' or not response_correct(
                (folder/'functional/stdout.txt').read_text(), (folder/'functional/stderr.txt').read_text()):
            errors.append('natural-language arithmetic smoke test failed')
        natural = placement((folder/'functional/stderr.txt').read_text())
        errors += verify_placement(natural,policy,self.layout)
        diagnostic = None
        if not errors:
            diagnostic = self.launch(bench_command(self.config,policy,1,True),folder/'diagnostic',True)
            diag = placement((folder/'diagnostic/stderr.txt').read_text())
            errors += verify_placement(diag,policy,self.layout)
            if diagnostic['classification'] != 'ok':
                errors.append('diagnostic PP/TG execution failed')
            # Custom ratios and explicit offload counts should reproduce the source-derived map.
            if policy.tensor_split is not None or len(policy.devices)==1:
                expected = {str(i):d for d,ids in screening.estimated_layers.items() for i in ids}
                expected.update({str(i):'CPU' for i in range(self.layout.blocks+1) if str(i) not in expected})
                if diag['layers'] != expected:
                    errors.append('observed mapping differs from explicit requested layer policy')
            try:
                analysis.parse_benchmark(json.loads((folder/'diagnostic/stdout.txt').read_text()),self.config,1)
            except (ValueError,KeyError,TypeError) as e:
                errors.append(f'diagnostic benchmark invalid: {e}')
            write_json(folder/'diagnostic-placement.json',diag)
        value = {'status':'ok' if not errors else 'invalid','errors':errors,'validation_key':key,
                 'policy':policy.to_dict(),'functional_placement':natural,'directory':str(folder),
                 'note':'diagnostic timing excluded; quiet timed mapping is not directly traced'}
        write_json(folder/'validation.json',value)
        if not errors:
            self.validations[policy.name] = value
        return value

    def benchmark(self, policy, phase, reps, round_index=None):
        validation = self.validations.get(policy.name)
        if not validation or validation['validation_key'] != validation_key(self.config,policy,self.model,self.runtime_files):
            raise ValueError('matching successful functional/placement validation is required')
        folder = unique_directory(self.directory,policy.name+'-'+phase)
        preflight = discover(self.config)
        screening = estimate(policy,self.layout,preflight,self.config)
        write_json(folder/'preflight.json',{'snapshot':preflight,'screening':screening.to_dict()})
        if not screening.admissible:
            raise StopSuite(f'resources/identity changed: {screening.reasons}')
        process = self.launch(bench_command(self.config,policy,reps),folder/'process')
        record = {'run_id':folder.name,'directory':str(folder),'policy':policy.to_dict(),'phase':phase,
                  'round':round_index,'comparison_key':self.comparison_key,'validation_key':validation['validation_key'],
                  'validation_directory':validation['directory'],'status':'invalid','metrics':None,'process':process}
        if process['classification']=='ok':
            try:
                rows = json.loads((folder/'process/stdout.txt').read_text())
                # Check requested-policy identity as well as workload identity.
                for row in rows:
                    if row.get('n_gpu_layers') != policy.gpu_layers or row.get('model_filename') != self.config.model:
                        raise ValueError('model or requested layer count differs')
                    if row.get('devices') != '/'.join(policy.devices):
                        raise ValueError('requested devices differ in benchmark JSON')
                    ratios = [float(v) for v in row.get('tensor_split','').split('/')]
                    expected = list(policy.tensor_split) if policy.tensor_split else [0.0]
                    while len(ratios)>len(expected) and ratios[-1]==0:
                        ratios.pop()
                    if ratios != expected:
                        raise ValueError('requested tensor proportions differ in benchmark JSON')
                record['metrics'] = analysis.parse_benchmark(rows,self.config,reps)
                record['status'] = 'ok'
            except (ValueError,KeyError,TypeError) as e:
                record['error'] = str(e)
        write_json(folder/'record.json',record)
        self.records.append(record)
        return record

    def tune(self):
        choices = candidates(self.layout,self.snapshot,self.config)
        write_json(self.directory/'candidates.json',[c.to_dict() for c in choices])
        try:
            for c in choices:
                if c.admissible:
                    valid = self.validate(c.policy)
                    if valid['status']=='ok':
                        self.benchmark(c.policy,'explore',self.config.exploratory_reps)
            names = analysis.finalists(self.records)
            write_json(self.directory/'finalists.json',{'names':names,'order':balanced_order(names,self.config.final_rounds)})
            by_name = {c.policy.name:c.policy for c in choices}
            for r,order in enumerate(balanced_order(names,self.config.final_rounds)):
                for name in order:
                    self.benchmark(by_name[name],'final',self.config.final_reps,r)
            result = analysis.selection(self.records,self.config.final_rounds)
            result['records'] = self.records
            if result['selected']:
                result['selected_policy'] = by_name[result['selected']].to_dict()
            write_json(self.directory/'selection.json',result)
            return result
        except BaseException as e:
            write_json(self.directory/'stopped.json',{'reason':str(e),'launches':self.count,'records':self.records})
            raise
