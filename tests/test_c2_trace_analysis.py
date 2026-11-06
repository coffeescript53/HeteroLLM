"""Synthetic CPU-only trace semantics; no runtime or GPU invocation."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('c2_trace_analysis', Path(__file__).resolve().parents[1]/'scripts/analyze-c2-trace.py')
trace = importlib.util.module_from_spec(spec); spec.loader.exec_module(trace)


def fixture():
    header = dict(type='header',schema='heterollm.cpu_trace.v1',clock='steady_clock_ns',
                  time_domain='cpu',phase='unknown',capacity=65536,record_bytes=176)
    events = []
    for kind, start, end, src, dst, size in [
        ('copy.scheduler_fallback',1,101,'CUDA0','ROCm0',16),
        ('backend.synchronize',2,31,'CUDA0','',0),
        ('event.synchronize',20,41,'ROCm0','',0),
        ('copy.host_stage',41,101,'CUDA0','ROCm0',16),
        ('copy.host_stage.get',41,61,'CUDA0','host_malloc',16),
        ('copy.host_stage.set',61,101,'host_malloc','ROCm0',16),
        ('copy.host_stage',101,151,'ROCm0','CUDA0',32)]:
        events.append(dict(type='span',seq=len(events),thread=1,begin_ns=start,end_ns=end,
                           graph=1,split=1,copy=0,n_copies=4,logical_bytes=size,kind=kind,src=src,dst=dst))
    return [header,*events,dict(type='footer',recorded=len(events),dropped=0,unfinished=0,metadata_truncated=0,complete=True)]


class TraceAnalysisTests(unittest.TestCase):
    def run_rows(self, rows, stderr=''):
        return trace.analyze('\n'.join(map(json.dumps,rows)),stderr)

    def test_nested_wait_union_and_direction_counts(self):
        result = self.run_rows(fixture())
        self.assertEqual(result['cpu_wait_union_ns'],39)
        self.assertEqual(result['host_staged_directions']['CUDA0 -> ROCm0']['logical_bytes'],16)
        self.assertEqual(result['host_staged_directions']['CUDA0 -> ROCm0']['copies'],1)
        self.assertEqual(result['host_staged_directions']['ROCm0 -> CUDA0']['logical_bytes'],32)
        self.assertIsNone(result['gpu_execution_time_ns']); self.assertIsNone(result['gpu_overlap_ns'])
        self.assertEqual(result['phase'],'unknown')

    def test_incomplete_overflow_and_stderr_fail(self):
        rows = fixture()
        for field,value in [('complete',False),('dropped',1),('unfinished',1),('metadata_truncated',1),('recorded',8)]:
            changed = fixture(); changed[-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.run_rows(changed)
        with self.assertRaises(ValueError): self.run_rows(rows[:-1])
        with self.assertRaises(ValueError): self.run_rows(rows,'HETEROLLM_TRACE_ERROR: flush')

    def test_no_fabricated_phase_or_gpu_time(self):
        for field,value in [('phase','PP512'),('time_domain','gpu'),('capacity',70000)]:
            rows = fixture(); rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.run_rows(rows)

    def test_invalid_span_and_duplicate_fields(self):
        for field,value in [('seq',2),('end_ns',0),('logical_bytes',-1),('begin_ns',True),('end_ns',float('nan'))]:
            rows=fixture(); rows[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): self.run_rows(rows)
        with self.assertRaises(ValueError): trace.strict_json('{"x":1,"x":2}')

    def test_union_separate_nested_concurrent_and_bad_intervals(self):
        self.assertEqual(trace.union_ns([(10,20),(12,15),(18,25),(30,35)]),20)
        self.assertEqual(trace.union_ns([]),0)
        with self.assertRaises(ValueError): trace.union_ns([(2,1)])


if __name__ == '__main__': unittest.main()
