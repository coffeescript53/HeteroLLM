import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('c2_proposal',Path(__file__).resolve().parents[1]/'scripts/prepare-c2-proposal.py')
proposal=importlib.util.module_from_spec(spec);spec.loader.exec_module(proposal)


class C2ProposalTests(unittest.TestCase):
    def test_proposal_is_one_nonexecuting_process_with_frozen_controls(self):
        p=proposal.proposal();a=p['argv']
        self.assertFalse(p['execution_enabled']);self.assertEqual(p['maximum_model_processes'],1)
        self.assertIsNone(p['candidate_binary_sha256']);self.assertTrue(p['blocking_gates'])
        for key,value in {'-lm':'none','-lzm':'off','-ts':'1/1','-ngl':'999',
                          '-r':'1','-p':'512','-n':'128','-d':'0','-b':'512','-ub':'128'}.items():
            self.assertEqual(a[a.index(key)+1],value)
        self.assertNotIn('-v',a);self.assertNotIn('GGML_SCHED_DEBUG',p['environment'])
        self.assertEqual(p['limits']['rss_cap'],32*1024**3)
        self.assertEqual(p['limits']['admission_ram'],40*1024**3)
        self.assertEqual(p['limits']['swap_growth'],256*1024**2)
        self.assertEqual(p['limits']['temperature'],85)
        self.assertEqual(p['host_thermal']['policy']['dimm']['stop_c'],60)

    def test_source_binding_changes_with_proposal(self):
        p=proposal.proposal();first=proposal.manifest(p)
        p['argv'][p['argv'].index('-lm')+1]='mmap'
        self.assertNotEqual(first['plan_sha256'],proposal.manifest(p)['plan_sha256'])
        self.assertIn('heterollm/supervisor.py',first['files'])
        self.assertIn('instrumentation/ggml-cpu-trace.patch',first['files'])
        self.assertIn('llama.cpp/ggml/src/ggml-backend.cpp',first['files'])


if __name__=='__main__':unittest.main()
