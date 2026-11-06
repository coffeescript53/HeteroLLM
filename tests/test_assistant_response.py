"""Privacy-sanitized 70B output replay only; never launches a model."""
import gzip
import hashlib
import json
from pathlib import Path
import unittest

from heterollm.pilot import validate
from heterollm.runtime import PROMPT, response_correct

FIXTURES = Path(__file__).parent / 'fixtures'
STDOUT = (FIXTURES / 'stage_b_96gb_success_stdout.txt').read_text()
STDERR = gzip.decompress((FIXTURES / 'stage_b_96gb_success_stderr.txt.gz').read_bytes()).decode()
FINAL = '0.54.941.775 D Parsed message: {"role":"assistant","content":"42"}'
PREFIX, _, TAIL = STDOUT.partition('\n> ' + PROMPT + '\n')
FOOTER = '\n\n\nExiting...\n'


class AssistantResponseTests(unittest.TestCase):
    def test_exact_public_streams_and_pilot_validation(self):
        # C4 public derivatives normalize private paths. Original archived
        # stdout: 553697f47a13e8fd3cf363e1c9a115736e733fe5f629eb4682581b9de6e067ff
        # stderr: 4006fbfdc30c337a9b93fe5868142bbc122c37b3621b8b1b28157c259ec16ae3
        self.assertEqual(hashlib.sha256(STDOUT.encode()).hexdigest(),
                         'b962ffea5c6da1ecbfd9c768ccdbe0f7555468a55557bfd970dbdaee632a97e7')
        self.assertEqual(hashlib.sha256(STDERR.encode()).hexdigest(),
                         '05b0fb348e05cf6eab7e63ef4be4320aa791f5a19b20a6f7611037edfd5fa017')
        self.assertIn(FINAL, STDERR)
        self.assertFalse(response_correct(STDOUT))  # UI alone is insufficient.
        self.assertTrue(response_correct(STDOUT, STDERR))
        self.assertEqual(validate(STDOUT, STDERR)['errors'], [])

    def test_complete_answer_not_an_isolated_number(self):
        for answer in ('43', '', '42\nExplanation', '42\n43', '42\n42',
                       'The answer is 42', '42!', '42\x00', '\x1b[32m42',
                       '\u00a042', '42\v', '42<|eot_id|>'):
            with self.subTest(answer=answer):
                out = PREFIX + '\n> ' + PROMPT + '\n' + answer + FOOTER
                err = FINAL.split('Parsed message: ')[0] + 'Parsed message: ' + json.dumps(
                    {'role': 'assistant', 'content': answer})
                self.assertFalse(response_correct(out, err))

    def test_contradictory_stdout_or_final_message(self):
        self.assertFalse(response_correct(STDOUT.replace(TAIL, '43' + FOOTER), STDERR))
        self.assertFalse(response_correct(STDOUT, STDERR.replace(FINAL, FINAL.replace('"42"', '"43"'))))
        self.assertFalse(response_correct('42', FINAL.replace('"42"', '"43"')))
        self.assertFalse(response_correct('43', FINAL))

    def test_missing_partial_duplicate_or_extra_turn_message(self):
        for err in ('', STDERR.replace(FINAL, ''), FINAL + '\n' + FINAL,
                    FINAL + '\n' + FINAL.replace('"42"', '"43"'),
                    FINAL.replace('Parsed message:', 'Partial message:')):
            with self.subTest(err=err[:90]):
                self.assertFalse(response_correct(STDOUT, err))

    def test_malformed_or_ambiguous_json_fails_closed(self):
        prefix = FINAL.split('{')[0]
        for payload in ('{', '{"role":"assistant","content":"42"} trailing',
                        '{"role":"assistant","content":"43","content":"42"}',
                        '{"role":"user","content":"42"}', '["42"]', 'null',
                        '{"role":"assistant","content":42}',
                        '{"role":"assistant","content":"42","reasoning_content":"wrong"}',
                        '{"role":"assistant","content":"42","tool_calls":[]}'):
            with self.subTest(payload=payload):
                self.assertFalse(response_correct(STDOUT, prefix + payload))
        self.assertFalse(response_correct(STDOUT, 'untrusted prefix ' + FINAL))

    def test_unknown_or_incomplete_ui_is_rejected(self):
        for out in (STDOUT[:-1], STDOUT + 'extra', STDOUT.replace(FOOTER, ''),
                    STDOUT.replace(PROMPT, 'Different prompt'), STDOUT + STDOUT,
                    STDOUT.replace('b11434-5e03bdd87', 'unknown-build'),
                    STDOUT.replace('available commands:', 'unexpected framing:'),
                    STDOUT.replace('42' + FOOTER, '42\n[END]' + FOOTER)):
            with self.subTest(out=out[-80:]):
                self.assertFalse(response_correct(out, STDERR))

    def test_plain_response_contract_is_preserved(self):
        self.assertTrue(response_correct(' \t42\r\n'))
        self.assertTrue(response_correct('42', FINAL))
        self.assertFalse(response_correct('42\nExplanation', FINAL))


if __name__ == '__main__':
    unittest.main()
