"""Linux one-child supervisor. Internal CPU fixtures never imply GPU approval."""
from dataclasses import asdict, dataclass
import ctypes
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import stat
import time
import traceback

from .hardware import sha256_file
from .host_thermal import ThermalGuard, sample as host_thermal_sample
from .runner import classify

MIB = 1024**2


MONITOR_ATOMIC_TARGETS = frozenset(
    name + '-latest.json' + suffix
    for name in ('preflight', 'preflight-gpu', 'fast', 'gpu', 'maps', 'thermal')
    for suffix in ('', '.busy')
)


def evidence_sizes(directory):
    """Sample regular-file sizes; only a recognized atomic rename may vanish.

    Avoid is_file()/stat() TOCTOU. If a monitor temporary disappears, account
    for its published destination even if enumeration missed that new name.
    Keep the larger observation when a destination was already counted.
    Existing temporary files still count. All other filesystem errors escape.
    This is a live sample, not an atomic snapshot of concurrently growing logs.
    """
    sizes = {}
    for path in Path(directory).iterdir():
        name = path.name
        try:
            info = path.stat()
        except FileNotFoundError:
            if not name.endswith('.tmp') or name[:-4] not in MONITOR_ATOMIC_TARGETS:
                raise
            name = name[:-4]
            info = path.with_name(name).stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('atomic monitor destination is not a regular file: ' + name)
        if stat.S_ISREG(info.st_mode):
            sizes[name] = max(sizes.get(name, 0), info.st_size)
    return sizes


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_json(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def stamp():
    return {'monotonic': time.monotonic(), 'utc_ns': time.time_ns()}


def death_signal(parent):
    # Also stop an owned process if the recorder itself is SIGKILLed.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG')
    if os.getppid() != parent:
        os._exit(125)


def group_signal(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def live_group(pid):
    live = []
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = p.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pid and fields[0] not in ('Z', 'X'):
                live.append(int(p.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return live


def _reader(fn, path, period, parent, wake):
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    death_signal(parent)
    seq = 0
    while True:
        started = stamp()
        atomic_json(str(path) + '.busy', started)
        try:
            data = fn()
            error = None
        except BaseException as e:
            # Sensor exceptions retain raw readings even though the sample is rejected.
            data, error = getattr(e, 'evidence', None), f'{type(e).__name__}: {e}'
            error_traceback = traceback.format_exc()
        seq += 1
        value = {'seq': seq, 'started': started, 'ended': stamp(), 'data': data, 'error': error}
        if error:
            value['traceback'] = error_traceback
        atomic_json(path, value)
        if error or period is None:
            return
        wake.wait(max(0, period - (time.monotonic() - started['monotonic'])))
        wake.clear()


class Reader:
    """Separate process: a stuck driver/proc query cannot block the safety loop."""
    def __init__(self, name, fn, directory, period, budget):
        self.name, self.budget, self.seq = name, budget, 0
        self.path = Path(directory) / (name + '-latest.json')
        self.created = time.monotonic()
        self.wake = mp.get_context('fork').Event()
        self.process = mp.get_context('fork').Process(target=_reader,
            args=(fn, self.path, period, os.getpid(), self.wake))
        self.process.start()

    def poll(self):
        now = time.monotonic()
        busy = json.loads(Path(str(self.path) + '.busy').read_text()) if Path(str(self.path) + '.busy').exists() else None
        value = json.loads(self.path.read_text()) if self.path.exists() else None
        in_flight = busy and (not value or busy['monotonic'] > value['ended']['monotonic'])
        if in_flight and now - busy['monotonic'] > self.budget:
            raise RuntimeError(self.name + ': read timeout')
        if not value and now - self.created > self.budget:
            raise RuntimeError(self.name + ': startup timeout')
        if value and value['seq'] != self.seq:
            self.seq = value['seq']
            append_json(self.path.parent / (self.name + '.jsonl'), value)
            if value['error']:
                raise RuntimeError(self.name + ': ' + value['error'])
            if value['ended']['monotonic'] - value['started']['monotonic'] > self.budget:
                raise RuntimeError(self.name + ': slow sample')
            return value
        if not self.process.is_alive() and (not value or value['error']):
            raise RuntimeError(self.name + ': reader exited')
        return None

    def close(self):
        # Readers and any helper queries have separate owned sessions.
        if self.process.is_alive():
            group_signal(self.process.pid, signal.SIGKILL)
            self.process.kill()  # handles the narrow pre-setsid startup window
        self.process.join(timeout=.25)
        return not self.process.is_alive()


@dataclass(frozen=True)
class Limits:
    overall: float = 900
    preflight: float = 120
    term_grace: float = 2
    ram_floor: int = 4096*MIB
    rss_cap: int = 8192*MIB
    swap_growth: int = 256*MIB
    temperature: float = 85
    log_cap: int = 256*MIB
    evidence_cap: int = 512*MIB
    admission_ram: int = 16384*MIB
    admission_temperature: float = 85


def guard(fast, gpu, maps, baseline, limits):
    if fast['memory']['MemAvailable'] < limits.ram_floor:
        return 'host_memory_pressure'
    if fast['swap_used'] - baseline > limits.swap_growth:
        return 'swap_growth'
    if max(fast['process']['VmRSS'], fast['process']['VmHWM']) > limits.rss_cap:
        return 'process_rss_limit'
    if fast['memory'].get('HardwareCorrupted', 0):
        return 'hardware_memory_error'
    if max(g['temperature_c'] for g in gpu.values()) >= limits.temperature:
        return 'temperature_limit'
    if maps['model_mappings']:
        return 'unexpected_model_mapping'
    return None


def admission_reason(fast, gpu, preflight, limits):
    """Recheck capacity immediately before exec, without crediting freed VRAM."""
    if fast['memory']['MemAvailable'] < limits.admission_ram:
        return 'admission_ram_changed'
    if any(g['temperature_c'] > limits.admission_temperature for g in gpu.values()):
        return 'admission_temperature'
    baseline = preflight.get('capacity_baseline', {})
    for device, size in preflight.get('screening', {}).get('gpu_estimated_bytes', {}).items():
        reading = gpu[device]
        free = min(reading['total_bytes'] - reading['used_bytes'],
                   reading.get('free_bytes', reading['total_bytes'] - reading['used_bytes']))
        if device in baseline:
            # CUDA runtime free memory can be below NVML total-used. Retain that
            # stricter ceiling, subtracting any observed increase since admission.
            before = baseline[device]
            free = min(free, before['runtime_free_bytes'] -
                       max(0, reading['used_bytes'] - before['sensor_used_bytes']))
        if free < size + 3072*MIB:
            return 'admission_vram_changed'
    return None


class LoadingModeGuard:
    """Check complete byte lines; a temporary EOF while the child lives is not final."""
    marker = re.compile(rb'\bload_mode\b')
    approved = re.compile(rb'\(load_mode[ \t]*=[ \t]*none[ \t]*\)[ \t]*\r?\Z')
    standalone = re.compile(rb'[ \t]*load_mode[ \t]*=[ \t]*none[ \t]*\r?\Z')

    def __init__(self):
        self.pending = b''
        self.rejected_line = None

    def feed(self, chunk, *, final=False):
        if self.rejected_line is not None:
            return False
        lines = (self.pending + chunk).split(b'\n')
        self.pending = lines.pop()
        if final and self.pending:
            lines.append(self.pending)  # Definitive EOF: this record cannot grow.
            self.pending = b''
        for line in lines:
            markers = self.marker.findall(line)
            if markers and (len(markers) != 1 or not (
                    self.approved.search(line) or self.standalone.fullmatch(line))):
                self.rejected_line = line
                return False
        return True

    def finish(self, path, offset):
        """After child exit/cleanup, check unread bytes and the last EOF-delimited line."""
        with Path(path).open('rb') as f:
            f.seek(offset)
            while chunk := f.read(256*1024):
                if not self.feed(chunk):
                    return False
        return self.feed(b'', final=True)


def supervise(argv, directory, environment, preflight, samplers, validate, limits=Limits(), *,
              scope='one functional process; no retry or benchmark'):
    """Exactly one fork/exec; never retries. Caller holds the execution lock.

    preflight is a bounded read-only callable returning a frozen swap baseline.
    samplers(pid) returns fast/GPU/maps callables. A mandatory host thermal
    reader uses the pinned physical sensors; only CPU tests inject its callable.
    The CLI exposes neither arbitrary argv nor limit overrides.
    """
    start = time.monotonic()
    directory = Path(directory)
    directory.mkdir(exist_ok=False, parents=True)
    events = directory / 'events.jsonl'
    result = {'status': 'incomplete', 'launches': 0, 'pid': None, 'reason': None,
              'cleanup_complete': False, 'started': stamp(), 'limits': asdict(limits)}
    pid = None; readers = []; reaped = False; gate = None; status = None
    interrupted = []
    previous = {s: signal.signal(s, lambda sig, frame: interrupted.append(sig))
                for s in (signal.SIGTERM, signal.SIGINT)}
    latest = {}; stop = None; stderr_offset = 0; tail = ''; phases = set(); oom_baseline = None
    mode_guard = LoadingModeGuard()
    thermal_guard = ThermalGuard()
    def event(kind, **fields):
        append_json(events, {'event': kind, **stamp(), **fields})
    def reap():
        nonlocal reaped, status
        if pid and not reaped:
            got, status_raw, usage = os.wait4(pid, os.WNOHANG)
            if got:
                reaped = True; status = os.waitstatus_to_exitcode(status_raw)
                result['peak_child_rss_bytes'] = usage.ru_maxrss * 1024
                atomic_json(directory/'wait4-rusage.json', {'pid':pid, 'exit_code':status,
                    'units':{'ru_maxrss':'KiB', 'ru_minflt':'count', 'ru_majflt':'count'},
                    'rusage':{k:getattr(usage,k) for k in dir(usage) if k.startswith('ru_')}, **stamp()})
                event('reaped', exit_code=status)
    try:
        atomic_json(directory/'command.json', {'argv':argv, 'environment':environment, 'limits':asdict(limits)})
        atomic_json(directory/'authorization.json', {'scope':scope,
                                                     'invocation_gate':'explicit --execute required by public CLI'})
        atomic_json(directory/'supervisor-identity.json', {'path':__file__, 'sha256':sha256_file(__file__)})
        atomic_json(directory/'result.json', result)
        event('preflight_started')
        worker = Reader('preflight', preflight, directory, None, limits.preflight)
        readers.append(worker)
        pre = None
        while pre is None:
            if interrupted:
                raise InterruptedError('supervisor_signal')
            if time.monotonic()-start >= min(limits.preflight, limits.overall):
                raise TimeoutError('preflight_timeout')
            pre = worker.poll()
            time.sleep(.02)
        worker.close()
        baseline = pre['data']['swap_baseline']
        atomic_json(directory/'preflight.json', pre['data'])
        # Gate keeps the child from exec until parent monitoring is ready.
        read_gate, gate = os.pipe()
        event('launch_intent', argv=argv)
        parent = os.getpid()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(gate); os.setsid()
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                death_signal(parent)
                os.chdir(directory)
                for fd, name in ((0, '/dev/null'), (1, 'stdout.txt'), (2, 'stderr.txt')):
                    h = os.open(name, os.O_RDONLY if fd == 0 else os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
                    os.dup2(h, fd); os.close(h)
                if os.read(read_gate, 1) != b'1':
                    os._exit(125)
                os.close(read_gate)
                os.execve(argv[0], argv, environment)
            except BaseException as e:
                os.write(2, (f'exec failure: {type(e).__name__}: {e}\n').encode())
                os._exit(127)
        os.close(read_gate)
        result['pid'] = pid
        event('child_owned', pid=pid)
        funcs = samplers(pid)
        funcs['thermal'] = funcs.get('thermal', host_thermal_sample)
        monitoring = {}
        for name,period,budget in [('fast',.25,1),('gpu',1,2),('maps',2,1),('thermal',1,1)]:
            reader = Reader(name, funcs[name], directory, period, budget)
            readers.append(reader)
            monitoring[name] = reader
        released = False; process_identity = None
        while True:
            if interrupted:
                stop = 'supervisor_signal'; break
            if time.monotonic()-start >= limits.overall:
                stop = 'overall_timeout'; break
            if not released and thermal_guard.admission_expired(time.monotonic()):
                stop = 'host_thermal_admission_cooldown_timeout'; break
            # Check guards before consuming the exit, including final samples.
            for name, reader in monitoring.items():
                value = reader.poll()
                if value:
                    latest[name] = value
                    if name == 'thermal':
                        thermal_stop = thermal_guard.observe(value['data'], time.monotonic(), admission=not released)
                        atomic_json(directory/'thermal-summary.json', thermal_guard.summary())
                        for transition in thermal_guard.transitions:
                            event('host_thermal_state', **transition)
                        if thermal_stop:
                            stop = thermal_stop
                            break
                    if name == 'fast':
                        identity = value['data']['process']['start_ticks']
                        if process_identity is not None and identity != process_identity:
                            raise RuntimeError('process identity changed')
                        process_identity = identity
                        oom = value['data'].get('oom_kill',0)
                        if oom_baseline is None:
                            oom_baseline = oom
                        elif oom > oom_baseline:
                            raise RuntimeError('system OOM kill counter increased')
                if name in latest and time.monotonic()-latest[name]['started']['monotonic'] > {'fast':1,'gpu':3,'maps':3,'thermal':3}[name]:
                    raise RuntimeError(name + ': stale sample')
            if stop:
                break
            if len(latest) == 4:
                stop = guard(*(latest[n]['data'] for n in ('fast','gpu','maps')), baseline, limits)
                if stop:
                    break
                if not released:
                    stop = admission_reason(latest['fast']['data'], latest['gpu']['data'], pre['data'], limits)
                    if stop:
                        break
                    # Two fresh successful sensor reads establish readiness;
                    # all other safety and deadline checks continue while gated.
                    if thermal_guard.admission_safe_samples >= 2:
                        event('host_thermal_ready', summary=thermal_guard.summary())
                        event('exec_release', pid=pid)
                        os.write(gate, b'1'); os.close(gate); gate = None
                        result['launches'] = 1; released = True
            sizes = evidence_sizes(directory)
            if sum(sizes.get(n,0) for n in ('stdout.txt','stderr.txt')) > limits.log_cap:
                stop = 'log_limit'; break
            if sum(sizes.values()) > limits.evidence_cap:
                stop = 'evidence_limit'; break
            stderr_path = directory/'stderr.txt'
            if stderr_path.exists():
                with stderr_path.open('rb') as f:
                    f.seek(stderr_offset); chunk = f.read(256*1024); stderr_offset += len(chunk)
                text = tail + chunk.decode(errors='replace'); tail = text[-4096:]
                for phase in ('load_mode', 'model buffer size', 'KV buffer size', 'n_ctx', 'MUL_MAT'):
                    if phase in text and phase not in phases:
                        phases.add(phase); event('loader_phase', phase=phase)
                        monitoring['maps'].wake.set()
                if not mode_guard.feed(chunk):
                    stop = 'unexpected_loading_mode'; break
                failure = classify(0, text)
                if failure != 'ok':
                    stop = failure; break
            reap()
            if reaped:
                stop = None if status == 0 else 'runtime_failure'
                break
            time.sleep(.02)
    except BaseException as e:
        stop = str(e) if isinstance(e, (TimeoutError, InterruptedError)) else 'monitor_or_supervisor_failure'
        result['error'] = f'{type(e).__name__}: {e}'
        result['traceback'] = traceback.format_exc()
    finally:
        try:
            if gate is not None:
                os.close(gate)
            if pid:
                event('cleanup_intent', reason=stop, pid=pid)
                if stop:
                    atomic_json(directory/'stop-intent.json', {
                        'reason':stop, 'latest':latest,
                        'error':result.get('error'), 'traceback':result.get('traceback'), **stamp()})
                group_signal(pid, signal.SIGTERM)
                until = time.monotonic() + limits.term_grace
                while time.monotonic() < until:
                    reap()
                    if reaped and not live_group(pid):
                        break
                    time.sleep(.02)
                group_signal(pid, signal.SIGKILL)
                event('kill_group_cleanup', pid=pid)
                until = time.monotonic() + 1
                while time.monotonic() < until:
                    reap()
                    if reaped and not live_group(pid):
                        break
                    time.sleep(.02)
                result['cleanup_complete'] = reaped and not live_group(pid)
            else:
                result['cleanup_complete'] = True
            result['reader_cleanup_complete'] = all([r.close() for r in readers])
            result['exit_code'] = status
            if not result['cleanup_complete'] or not result['reader_cleanup_complete']:
                stop = stop or 'cleanup_incomplete'
            if not stop and result.get('peak_child_rss_bytes', 0) > limits.rss_cap:
                stop = 'process_rss_limit'
                event('post_exit_rss_limit', peak_bytes=result['peak_child_rss_bytes'])
            if pid and not stop and not mode_guard.finish(directory/'stderr.txt', stderr_offset):
                stop = 'unexpected_loading_mode'
            if pid and not stop:
                stdout = (directory/'stdout.txt').read_text(errors='replace')
                stderr = (directory/'stderr.txt').read_text(errors='replace')
                result['validation'] = validate(stdout, stderr)
                if result['validation']['errors']:
                    stop = 'validation_failure'
            result['reason'] = stop
            result['status'] = 'passed' if pid and reaped and not stop else 'failed'
        except BaseException as e:
            # A failed evidence disk cannot be made durable by any software.
            if pid:
                group_signal(pid, signal.SIGKILL)
                reap()
            for reader in readers:
                reader.close()
            result.update(status='failed', reason='finalization_failure', finalization_error=str(e),
                          finalization_traceback=traceback.format_exc())
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            result['ended'] = stamp(); result['elapsed_seconds'] = time.monotonic()-start
            result['host_thermal'] = thermal_guard.summary()
            result['received_signals'] = interrupted
            result['raw_files'] = {p.name:{'size_bytes':p.stat().st_size, 'sha256':sha256_file(p)}
                                  for p in (directory/'stdout.txt',directory/'stderr.txt') if p.exists()}
            atomic_json(directory/'result.json', result)
            event('final', status=result['status'], reason=result['reason'])
    return result
