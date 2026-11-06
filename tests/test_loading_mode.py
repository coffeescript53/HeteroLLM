"""CPU-only replay: privacy-sanitized Stage B stderr, complete records/chunks."""
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

from heterollm.supervisor import LoadingModeGuard

REAL_STDERR = Path(__file__).parent/'fixtures/stage_b_96gb_stderr.txt'
REAL_LINE = b'0.00.484.188 I load_tensors: loading model tensors, this can take a while... (load_mode = none)\n'


class LoadingModeTests(unittest.TestCase):
    def test_exact_failed_line_at_every_split_and_one_byte_chunks(self):
        for split in range(len(REAL_LINE)+1):
            guard=LoadingModeGuard()
            self.assertTrue(guard.feed(REAL_LINE[:split]))
            self.assertTrue(guard.feed(REAL_LINE[split:]))
            self.assertTrue(guard.feed(b'',final=True))
        guard=LoadingModeGuard()
        for byte in REAL_LINE:self.assertTrue(guard.feed(bytes([byte])))
        self.assertEqual(guard.pending,b'')

    def test_no_decision_on_partial_lines_or_temporary_eof(self):
        for suffix in (b'n',b'none',b'none)',b'mmap)',b'dio)',b'garbage)',b''):
            guard=LoadingModeGuard()
            partial=b'loader: (load_mode = '+suffix
            self.assertTrue(guard.feed(partial))
            self.assertTrue(guard.feed(b''))  # Tail EOF while writer still runs.
            self.assertIsNone(guard.rejected_line)
            self.assertEqual(guard.feed(b'\n'),suffix==b'none)')

    def test_unapproved_and_malformed_complete_lines_at_every_split(self):
        values=(b'mmap',b'dio',b'auto',b'mlock',b'mmap+mlock',b'NONE',b'noneX',b'none mmap',
                b'none,',b'"none"',b'',b'none\x00',b'none\xff')
        lines=[REAL_LINE.replace(b'= none)',b'= '+v+b')') for v in values]
        lines += [b'load_mode\n',b'load_mode: none\n',b'(load_mode == none)\n',
                  b'(load_mode = none\n',b'load_mode = none)\n',b'(load_mode = none))\n',
                  b'(load_mode = none) trailing\n',b'(load_mode = mmap) (load_mode = none)\n']
        for line in lines:
            for split in range(len(line)):
                with self.subTest(line=line,split=split):
                    guard=LoadingModeGuard()
                    self.assertTrue(guard.feed(line[:split]))
                    self.assertFalse(guard.feed(line[split:]))
                    self.assertFalse(guard.feed(REAL_LINE))  # Failure is sticky.

    def test_crlf_tabs_standalone_and_multiple_records(self):
        guard=LoadingModeGuard()
        for line in (REAL_LINE.replace(b'\n',b'\r\n'),b'load_mode = none\n',
                     b'loader: (load_mode\t=\tnone\t)\r\n',b'other line\n'):
            self.assertTrue(guard.feed(line))
        self.assertFalse(guard.feed(REAL_LINE+b'(load_mode = dio)\n'+REAL_LINE))

    def test_definitive_eof_checks_unterminated_record(self):
        for line,valid in [(REAL_LINE[:-1],True),(b'(load_mode = mmap)',False),
                           (b'(load_mode = none',False),(b'load_mode = ',False)]:
            guard=LoadingModeGuard()
            self.assertTrue(guard.feed(line))
            self.assertEqual(guard.feed(b'',final=True),valid)

    def test_finish_checks_unread_tail_beyond_one_read_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'stderr.txt'
            for suffix,valid in [(REAL_LINE,True),(b'(load_mode = dio)\n',False),
                                 (b'(load_mode = none',False)]:
                data=b'unrelated\n'*70000+suffix;p.write_bytes(data)
                guard=LoadingModeGuard();offset=123
                self.assertTrue(guard.feed(data[:offset]))
                self.assertEqual(guard.finish(p,offset),valid)

    def test_full_real_stderr_replay_across_chunk_schedules(self):
        data=REAL_STDERR.read_bytes()
        # Public C4 derivative: private paths were normalized before release.
        # Original archived SHA-256 (unchanged in private evidence):
        # 0f54d1c1d2af99736016e480ababb79c658968c43a15b5a0410cad68c0944519
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         'bbe7d5840f61947c6dccf3d57d364513f25f1a1ead3a1987e8d63ad316b951cb')
        self.assertEqual(data.splitlines(keepends=True)[385],REAL_LINE)
        for size in (1,2,7,31,4096,256*1024):
            with self.subTest(size=size):
                guard=LoadingModeGuard()
                for pos in range(0,len(data),size):self.assertTrue(guard.feed(data[pos:pos+size]))
                self.assertTrue(guard.feed(b'',final=True))
        rng=random.Random(42);guard=LoadingModeGuard();pos=0
        while pos<len(data):
            size=rng.randint(1,8192)
            self.assertTrue(guard.feed(data[pos:pos+size]));pos+=size
        self.assertTrue(guard.feed(b'',final=True))


if __name__ == '__main__':unittest.main()
