"""CPU-only preflight tests: all runtime, model and sensor I/O is mocked."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from heterollm import stage_c as c0
from helpers import config
from test_stage_c import hardware, LAYOUT, GIB
from test_supervisor import gpu_fixture

ORIGINAL = 98774167552
MIB = 1024**2


class MemoryIdentityTests(unittest.TestCase):
    def preflight(self, observed=ORIGINAL, *, snapshot_change=None,
                  signature_change=None, controls=None, available=64*GIB,
                  expected_error=None):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            directory = Path(temp)
            conf = config(output=temp)
            snap = hardware()
            baseline = {'signature': c0.hardware_signature(snap), 'controls': {'control': 'auto'}}
            snap['telemetry']['memory'].update(MemTotal=observed, MemAvailable=available)
            if snapshot_change:
                snapshot_change(snap)
            signature = c0.hardware_signature(snap)
            if signature_change:
                signature_change(signature)
            for name, value in [('runtime.json', {}), ('deps.json', {'files': {}}),
                                ('baseline.json', baseline)]:
                (directory/name).write_text(json.dumps(value))
            plan = {'receipt': 'synthetic', 'runtime_baseline': str(directory/'runtime.json'),
                    'dependency_baseline': str(directory/'deps.json'),
                    'hardware_baseline': str(directory/'baseline.json'), 'mem_total_bytes': ORIGINAL}
            original_inputs = deepcopy((snap, baseline, plan))
            replacements = {'load_receipt': {}, 'runtime_identity': {}, 'dependencies': {'files': {}},
                            'concurrent_runtimes': [], 'discover': snap,
                            'host_controls': baseline['controls'] if controls is None else controls,
                            'hardware_signature': signature, 'read_layout': LAYOUT,
                            'fast_sample': {}, 'maps_sample': {}, 'bounded_gpu_readiness': gpu_fixture()}
            for name, value in replacements.items():
                stack.enter_context(patch.object(c0, name, return_value=value))
            stack.enter_context(patch.object(c0, 'IDLE_SECONDS', 0))
            stack.enter_context(patch.object(c0.shutil, 'disk_usage',
                                            return_value=type('Disk', (), {'free': 20*GIB})()))
            if expected_error:
                with self.assertRaisesRegex(ValueError, expected_error):
                    c0.preflight(conf, plan, c0.POLICIES[0], 0)
            else:
                c0.preflight(conf, plan, c0.POLICIES[0], 0)
            self.assertEqual((snap, baseline, plan), original_inputs)
            self.assertEqual(json.loads((directory/'baseline.json').read_text()), baseline)
            return json.loads((directory/'memory-identity.json').read_text())

    def assert_record(self, record, observed, accepted):
        self.assertEqual(record, {'original_bytes': ORIGINAL, 'observed_bytes': observed,
                                 'signed_difference_bytes': observed-ORIGINAL,
                                 'tolerance_bytes': MIB, 'accepted': accepted,
                                 'decision': 'accept' if accepted else 'reject'})

    def test_both_checks_accept_observed_reboot_drift_and_inclusive_boundaries(self):
        # 98,774,179,840 is the exact MemTotal saved on 25 October (+12 KiB).
        for observed in (ORIGINAL, 98774179840, ORIGINAL-12288, ORIGINAL-1,
                         ORIGINAL+MIB, ORIGINAL-MIB):
            with self.subTest(observed=observed):
                records = self.preflight(observed)
                for record in records.values():
                    self.assert_record(record, observed, True)

    def test_hardware_check_rejects_beyond_tolerance_and_missing_dimm_scale_loss(self):
        for delta in (MIB+1, -MIB-1, 2*MIB, -48*GIB):
            with self.subTest(delta=delta):
                records = self.preflight(ORIGINAL+delta, expected_error='identity changed')
                for record in records.values():
                    self.assert_record(record, ORIGINAL+delta, False)

    def test_pinned_plan_check_independently_rejects_out_of_range_observation(self):
        # Isolate the second guard with a matching synthetic signature; the
        # actual memory snapshot still reaches the independent pinned-plan check.
        for delta in (MIB+1, -MIB-1, -48*GIB):
            with self.subTest(delta=delta):
                records = self.preflight(ORIGINAL+delta,
                    signature_change=lambda s: s.update(mem_total_bytes=ORIGINAL),
                    expected_error='>=40 GiB')
                self.assert_record(records['hardware_baseline'], ORIGINAL, True)
                self.assert_record(records['pinned_plan'], ORIGINAL+delta, False)

    def test_hardware_check_independently_rejects_signature_drift(self):
        for delta in (MIB+1, -MIB-1):
            with self.subTest(delta=delta):
                records = self.preflight(signature_change=lambda s: s.update(mem_total_bytes=ORIGINAL+delta),
                                         expected_error='identity changed')
                self.assert_record(records['hardware_baseline'], ORIGINAL+delta, False)
                self.assert_record(records['pinned_plan'], ORIGINAL, True)

    def test_40_gib_available_ram_requirement_is_unchanged(self):
        self.preflight(98774179840, available=40*GIB)
        records = self.preflight(98774179840, available=40*GIB-1, expected_error='>=40 GiB')
        self.assertTrue(all(r['accepted'] for r in records.values()))

    def test_all_other_signature_fields_and_keys_remain_exact(self):
        mutations = [lambda s: s.update(platform='other'), lambda s: s['cpu'].append('changed'),
                     lambda s: s.update(swap_total_bytes=s['swap_total_bytes']+1),
                     lambda s: s['devices'][0].update(pci='0000:06:00.0'),
                     lambda s: s['devices'][1].update(name='ROCm1'),
                     lambda s: s['devices'][1].update(description='other'),
                     lambda s: s['nvidia_drivers'][0].update(driver='other'),
                     lambda s: s.pop('cpu'), lambda s: s.update(unapproved_field=True)]
        for i, change in enumerate(mutations):
            with self.subTest(change=i):
                records = self.preflight(98774179840, signature_change=change,
                                         expected_error='identity changed')
                self.assertTrue(all(r['accepted'] for r in records.values()))

    def test_configured_controls_remain_exact(self):
        for controls in ({'control': 'on'}, {}, {'control': 'auto', 'extra': 'value'}):
            with self.subTest(controls=controls):
                self.preflight(98774179840, controls=controls, expected_error='identity changed')

    def test_missing_or_invalid_observations_are_rejected_and_recorded(self):
        for observed in (None, True, '98774167552', float(ORIGINAL), 0, -1):
            with self.subTest(observed=observed):
                records = self.preflight(observed, expected_error='identity changed')
                for record in records.values():
                    self.assertFalse(record['accepted'])
                    self.assertEqual(record['decision'], 'reject')
                    self.assertIsNone(record['signed_difference_bytes'])
        records = self.preflight(snapshot_change=lambda s: s['telemetry']['memory'].pop('MemTotal'),
                                 expected_error='identity changed')
        self.assertIsNone(records['hardware_baseline']['observed_bytes'])

    def test_invalid_original_never_becomes_a_new_baseline(self):
        for original in (None, True, str(ORIGINAL), float(ORIGINAL), 0, -1):
            self.assertFalse(c0.mem_total_comparison(original, ORIGINAL)['accepted'])


if __name__ == '__main__':
    unittest.main()
