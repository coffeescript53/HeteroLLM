"""CPU/filesystem only: deterministic rename races and real atomic writers."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from heterollm.supervisor import MONITOR_ATOMIC_TARGETS, atomic_json, evidence_sizes


def rename_during_stat(directory, name='preflight-gpu-latest.json.tmp'):
    """Force a real rename after enumeration and immediately before stat."""
    temporary = Path(directory)/name
    temporary.write_text('{"forced_race": true}\n')
    original_stat = Path.stat
    triggered = []

    def stat(path, *args, **kwargs):
        if path == temporary and not triggered:
            triggered.append(True)
            os.replace(path, path.with_name(name[:-4]))
        return original_stat(path, *args, **kwargs)

    with patch.object(Path, 'stat', stat):
        sizes = evidence_sizes(directory)
    assert triggered, 'race fixture was not exercised'
    return sizes


class EvidenceSizeTests(unittest.TestCase):
    def test_every_recognized_atomic_monitor_rename_is_accounted(self):
        for target in MONITOR_ATOMIC_TARGETS:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                sizes = rename_during_stat(temp, target+'.tmp')
                self.assertEqual(sizes, {target:len('{"forced_race": true}\n')})

    def test_new_destination_is_counted_when_enumeration_only_saw_temporary(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);temporary=directory/'fast-latest.json.tmp'
            with patch.object(Path,'iterdir',return_value=iter([temporary])):
                sizes=rename_during_stat(directory,'fast-latest.json.tmp')
            self.assertEqual(sizes, {'fast-latest.json':22})

    def test_renamed_destination_never_overwrites_larger_observed_size(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);dest=d/'fast-latest.json';dest.write_bytes(b'x'*4096)
            tmp=d/'fast-latest.json.tmp'
            with patch.object(Path,'iterdir',return_value=iter([dest,tmp])):
                sizes=rename_during_stat(d,tmp.name)
            self.assertEqual(sizes[dest.name],4096)

    def test_existing_temporaries_logs_and_other_regular_files_all_count(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);(d/'subdirectory').mkdir()
            expected={'fast-latest.json.tmp':17,'fast-latest.json':31,
                      'stdout.txt':5000,'stderr.txt':1000,'unexpected.tmp':100,
                      'fast.jsonl':200,'stop-intent.json':43}
            for name,size in expected.items():(d/name).write_bytes(b'x'*size)
            self.assertEqual(evidence_sizes(d),expected)

    def test_missing_permanent_or_unrecognized_tmp_raises(self):
        for name in ('stdout.txt','stderr.txt','fast.jsonl','fast-latest.json',
                     'gpu-latest.json.busy','other.tmp','result.json.tmp',
                     'fast-latest.json.tmp.tmp'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/name;p.write_text('evidence')
                original_stat=Path.stat
                def stat(path,*args,**kwargs):
                    if path==p:path.unlink()
                    return original_stat(path,*args,**kwargs)
                with patch.object(Path,'stat',stat),self.assertRaises(FileNotFoundError):
                    evidence_sizes(temp)

    def test_missing_or_nonregular_published_destination_raises(self):
        for directory_target in (False,True):
            with self.subTest(directory_target=directory_target),tempfile.TemporaryDirectory() as temp:
                d=Path(temp);p=d/'fast-latest.json.tmp';p.write_text('temporary')
                if directory_target:(d/'fast-latest.json').mkdir()
                original_stat=Path.stat
                def stat(path,*args,**kwargs):
                    if path==p:path.unlink()
                    return original_stat(path,*args,**kwargs)
                error=ValueError if directory_target else FileNotFoundError
                with patch.object(Path,'stat',stat),self.assertRaises(error):
                    evidence_sizes(d)

    def test_permission_io_and_directory_errors_are_not_suppressed(self):
        for name in ('fast-latest.json.tmp','stdout.txt'):
            for error in (PermissionError(errno.EACCES,'denied'),OSError(errno.EIO,'I/O')):
                with self.subTest(name=name,error=error),tempfile.TemporaryDirectory() as temp:
                    d=Path(temp);p=d/name;p.write_text('x');original_stat=Path.stat
                    def stat(path,*args,**kwargs):
                        if path==p:raise error
                        return original_stat(path,*args,**kwargs)
                    with patch.object(Path,'stat',stat),self.assertRaises(type(error)):
                        evidence_sizes(d)
        with self.assertRaises(FileNotFoundError):
            evidence_sizes('/nonexistent-heterollm-evidence-directory')

    def stress(self, grow_log=False):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);(d/'stdout.txt').write_bytes(b'x'*8192)
            barrier=threading.Barrier(5 if grow_log else 4);iterations=300
            def writer(name):
                barrier.wait(timeout=5)
                for i in range(iterations):
                    atomic_json(d/(name+'-latest.json.busy'),{'iteration':i})
                    atomic_json(d/(name+'-latest.json'),{'iteration':i,'payload':'x'*(1024+i)})
                return iterations*2
            def log_writer():
                barrier.wait(timeout=5)
                with (d/'stdout.txt').open('ab',buffering=0) as log:
                    for _ in range(iterations):
                        log.write(b'x'*2048)
                        time.sleep(.0001)
                return 0
            start=time.monotonic();scans=0
            last_log_size=8192
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(writer,n) for n in ('fast','gpu','maps')]
                if grow_log:futures.append(pool.submit(log_writer))
                barrier.wait(timeout=5)
                while not all(f.done() for f in futures):
                    self.assertLess(time.monotonic()-start,15,'bounded stress test timed out')
                    sizes=evidence_sizes(d);scans+=1
                    self.assertGreaterEqual(sizes['stdout.txt'],last_log_size)
                    last_log_size=sizes['stdout.txt']
                    self.assertGreaterEqual(sum(sizes.values()),8192)
                    time.sleep(.0001)
                writes=sum(f.result() for f in futures)
            self.assertGreater(scans,0)
            self.assertEqual(evidence_sizes(d),{p.name:p.stat().st_size for p in d.iterdir()})
            self.assertEqual((d/'stdout.txt').stat().st_size,8192+(iterations*2048 if grow_log else 0))
            for name in ('fast','gpu','maps'):
                self.assertEqual(json.loads((d/(name+'-latest.json')).read_text())['iteration'],iterations-1)
            print(json.dumps({'stress':'concurrent_atomic_writers','writers':3,
                              'concurrent_log_writer':grow_log,
                              'atomic_writes':writes,'size_scans':scans,
                              'elapsed_seconds':time.monotonic()-start,'status':'passed'}),flush=True)

    def test_concurrent_atomic_writers_and_size_scans(self):
        self.stress()

    def test_concurrent_atomic_writers_and_growing_permanent_log(self):
        self.stress(grow_log=True)
