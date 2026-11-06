"""CPU-only arithmetic/matching regressions for the C3 offline report."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('archived', Path(__file__).with_name('analyze_archived_traces.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture():
    def span(kind, begin, end, src='', dst='', logical_bytes=0):
        return dict(kind=kind, begin_ns=begin, end_ns=end, src=src, dst=dst,
                    logical_bytes=logical_bytes, graph=1, split=2, thread=1)
    spans = [span('copy.scheduler_fallback', 20, 80, 'CUDA0', 'ROCm0', 32),
             span('backend.synchronize', 20, 50, 'CUDA0'),
             span('copy.host_stage', 50, 80, 'CUDA0', 'ROCm0', 32),
             span('copy.host_stage.get', 52, 60, 'CUDA0', 'host_malloc', 32),
             span('copy.host_stage.set', 60, 79, 'host_malloc', 'ROCm0', 32),
             span('graph.submit', 10, 20, 'CUDA0'), span('graph.submit', 80, 90, 'ROCm0')]
    def dev(i, kind, backend, begin, end, observed, logical_bytes=0):
        return dict(id=i, kind=kind, backend=backend, phase=1, graph=1, split=2,
                    thread=1, logical_bytes=logical_bytes, start_before=begin,
                    observed=observed, elapsed_ms=(end-begin)/1e6)
    devices = [dev(0, 'graph_replay', 'CUDA0', 11, 45, 49),
               dev(1, 'copy_d2h', 'CUDA0', 53, 58, 59, 32),
               dev(2, 'copy_h2d', 'ROCm0', 62, 77, 78, 32),
               dev(3, 'graph_replay', 'ROCm0', 81, 98, 99)]
    return spans, devices


def two_graph_fixture():
    spans, devices = fixture()
    second_spans = [dict(s, graph=2, begin_ns=s['begin_ns'] + 200,
                         end_ns=s['end_ns'] + 200) for s in spans]
    second_devices = [dict(d, id=d['id'] + 4, graph=2,
                           start_before=d['start_before'] + 200,
                           observed=d['observed'] + 200) for d in devices]
    return spans + second_spans, devices + second_devices


class AnalysisTests(unittest.TestCase):
    def summarize(self, spans=None, devices=None):
        ss, dd = fixture()
        return mod.summarize_phase('synthetic', 1, dict(name='tg128', begin=1, end=1000),
                                   ss if spans is None else spans, dd if devices is None else devices, 1001)

    def test_nested_union_is_not_sum(self):
        self.assertEqual(mod.union_ns([(10, 80), (20, 30), (70, 90), (100, 110)]), 90)

    def test_intersection_union_does_not_double_count(self):
        self.assertEqual(mod.intersection_union([(0, 100), (10, 80)], [(20, 60), (40, 90)]), 70)

    def test_invalid_interval_fails(self):
        with self.assertRaises(ValueError):
            mod.union_ns([(3, 2)])

    def test_phase_boundary_excludes_straddling_span(self):
        spans, _ = fixture()
        spans.append(dict(spans[0], begin_ns=0))
        self.assertEqual(len(mod.phase_spans(spans, {'begin': 1, 'end': 1000})), 7)

    def test_parent_count_bytes_and_nested_cpu_accounting(self):
        r = self.summarize()
        self.assertEqual(r['host_stage_count'], 1)
        self.assertEqual(r['host_stage_logical_bytes'], 32)
        self.assertEqual(r['host_stage_native_legs_count'], 2)
        self.assertEqual(r['cross_fallback_cpu_union_ms'], 60 / 1e6)
        self.assertEqual(r['fallback_nested_sync_union_ms'], 30 / 1e6)
        self.assertEqual(r['fallback_nested_stage_union_ms'], 30 / 1e6)
        self.assertEqual(r['fallback_uncovered_cpu_ms'], 0)

    def test_missing_native_leg_fails(self):
        _, dev = fixture()
        with self.assertRaises(ValueError):
            self.summarize(devices=[d for d in dev if d['kind'] != 'copy_d2h'])

    def test_duplicate_native_leg_fails(self):
        _, dev = fixture()
        dev.append(dict(dev[1], id=9))
        with self.assertRaises(ValueError):
            self.summarize(devices=dev)

    def test_wrong_bytes_fails(self):
        _, dev = fixture()
        dev[1]['logical_bytes'] = 64
        with self.assertRaises(ValueError):
            self.summarize(devices=dev)

    def test_cpu_clock_completion_order_can_be_proven(self):
        self.assertEqual(self.summarize()['source_completion_before_staging_before_destination_submit_count'], 1)

    def test_late_observation_is_unknown_not_fake_overlap(self):
        _, dev = fixture()
        dev[0]['observed'] = 75
        self.assertEqual(self.summarize(devices=dev)['source_completion_before_staging_before_destination_submit_count'], 0)

    def test_zero_copy_negative_finding_preserved(self):
        ss, dd = fixture()
        r = self.summarize(spans=[s for s in ss if not s['kind'].startswith('copy')],
                           devices=[d for d in dd if not d['kind'].startswith('copy')])
        self.assertEqual(r['host_stage_count'], 0)
        self.assertEqual(r['host_stage_logical_bytes'], 0)
        self.assertEqual(r['host_stage_native_legs_count'], 0)

    def test_duplicate_cpu_graph_backend_identity_rejected(self):
        spans, _ = fixture()
        spans.append(dict(spans[-1], thread=9, split=3))
        with self.assertRaisesRegex(ValueError, 'ambiguous graph/backend'):
            self.summarize(spans=spans)

    def test_duplicate_device_graph_backend_identity_rejected(self):
        _, devices = fixture()
        devices.append(dict(devices[-1], id=9, thread=9, split=3))
        with self.assertRaisesRegex(ValueError, 'ambiguous graph/backend'):
            self.summarize(devices=devices)

    def test_reverse_host_stage_direction_rejected(self):
        spans, _ = fixture()
        for s in spans:
            if s['kind'] == 'copy.host_stage':
                s.update(src='ROCm0', dst='CUDA0')
        with self.assertRaisesRegex(ValueError, 'unsupported host-stage direction'):
            self.summarize(spans=spans)

    def test_missing_graph_backend_rejected(self):
        _, devices = fixture()
        with self.assertRaisesRegex(ValueError, 'graph identities'):
            self.summarize(devices=[d for d in devices if d['id'] != 3])

    def test_adjacent_graph_completion_order_proven(self):
        spans, devices = two_graph_fixture()
        r = self.summarize(spans=spans, devices=devices)
        self.assertEqual(r['consecutive_graph_pairs'], 1)
        self.assertEqual(r['prior_rocm_completion_before_next_cuda_enqueue_count'], 1)
        self.assertEqual(r['source_completion_before_staging_before_destination_submit_count'], 2)

    def test_late_prior_graph_observation_leaves_adjacency_unknown(self):
        spans, devices = two_graph_fixture()
        devices[3]['observed'] = 250
        r = self.summarize(spans=spans, devices=devices)
        self.assertEqual(r['consecutive_graph_pairs'], 1)
        self.assertEqual(r['prior_rocm_completion_before_next_cuda_enqueue_count'], 0)
        self.assertEqual(r['source_completion_before_staging_before_destination_submit_count'], 2)


if __name__ == '__main__':
    unittest.main()
