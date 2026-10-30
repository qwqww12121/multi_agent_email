"""Synthetic unit/transport fixtures only; never inserted into production prompts."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'qa/deps'))
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from four_text_v4.contracts import ROLES, TASKS, RawEmail
from four_text_v4.contracts.domain import task_for
from four_text_v4.contracts.reports import shape, validate_report, strict_json
from four_text_v4.contracts.decoder_schema import decoder_schema
from four_text_v4.experts.base_prompt import BOS, report_prefix, parse_prompt
from four_text_v4.ingestion import parse_email
from four_text_v4.ingestion.input_budget import slice_input
from four_text_v4.scoring.render import render_report
from four_text_v4.scoring.json_transport import seeded_parser, generate_json_batch
from reference_runner import score_task


def fixture(role):
    report = shape(role)
    if role in ('action', 'persuasion'):
        key = 'requested_actions' if role == 'action' else 'influence_attempts'
        reason = 'no_actions_reason' if role == 'action' else 'no_influence_reason'
        report[key] = []
        report[reason] = dict(value='Informational.', status='inferred', scope='current',
                              evidence=[dict(field='body', quote='UNIT')])
    else:
        key = 'communication_context' if role == 'context' else 'main_topic'
        report[key] = dict(value='UNIT', status='observed', scope='current',
                           evidence=[dict(field='body', quote='UNIT')])
    return {key: report[key] for key in decoder_schema(role)['required']}


def prompt_for(role, source='kaggle', body='UNIT'):
    raw = RawEmail(id='unit', label_schema=source, source_dataset=source,
                   task_type=task_for(source), calibration_domain='unit', body=body)
    mail = parse_email(raw); chunk = slice_input(mail, 0, len(mail.body))
    return (raw, mail, chunk, role, render_report(mail, chunk, role))


def accepts(role, text, alphabet=''):
    parser = seeded_parser(role, '', alphabet)
    for char in text:
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


class ContractTests(unittest.TestCase):
    def test_hf_generate_masks_invalid_tokens_and_keeps_batched_roles_separate(self):
        # Exercise actual HF generate + actual LMFE with deterministic fake logits.
        # This tests transport, never Llama accuracy; no weights are downloaded.
        import string
        import types
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from transformers.modeling_outputs import CausalLMOutputWithPast
        torch.set_num_threads(1)
        symbols = [BOS, '<eos>', '<unk>', '<pad>'] + list(string.printable)
        vocab = {char:i for i,char in enumerate(symbols)}
        core = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
        core.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
        core.decoder = decoders.Fuse()
        tok = PreTrainedTokenizerFast(tokenizer_object=core, bos_token=BOS,
                    eos_token='<eos>', unk_token='<unk>', pad_token='<pad>')
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_embd=8, n_layer=1,
                    n_head=1, n_positions=32768, eos_token_id=tok.eos_token_id,
                    bos_token_id=tok.bos_token_id, pad_token_id=tok.pad_token_id)).eval()
        targets = [json.dumps(fixture(role), separators=(',', ':')) for role in ROLES]
        suffixes = [text[len(report_prefix(role)):] for role,text in zip(ROLES,targets)]
        token_rows = [tok.encode(text, add_special_tokens=False) for text in suffixes]
        calls = [0]

        def forward(this, input_ids=None, attention_mask=None, **kwargs):
            step = calls[0]; calls[0] += 1
            logits = torch.zeros((len(token_rows), 1, len(tok)))
            logits[:, :, tok.unk_token_id] = 100.  # forbidden special token wins without masking
            for i,row in enumerate(token_rows):
                target = row[step] if step < len(row) else tok.eos_token_id
                logits[i, 0, target] = 20.
            return CausalLMOutputWithPast(logits=logits)

        model.forward = types.MethodType(forward, model)
        backend = types.SimpleNamespace(tokenizer=tok, model=model, max_tokens=32768)
        answers = generate_json_batch(backend, [prompt_for(role)[-1] for role in ROLES],
                                      max(map(len, token_rows)) + 2)
        for i,(role, answer) in enumerate(zip(ROLES,answers)):
            self.assertEqual(answer['text'], targets[i])
            self.assertEqual(answer['raw_text'], suffixes[i])
            self.assertEqual(answer['generated_tokens'], len(token_rows[i]))
            self.assertEqual(answer['finish_reason'], 'stop')
            validate_report(strict_json(answer['text']), role, {'body':'UNIT'})

    def test_all_native_tasks_and_roles_are_completion_prompts(self):
        for source in TASKS:
            for role in ROLES:
                prompt = prompt_for(role, source)[-1]
                self.assertTrue(prompt.startswith(BOS))
                self.assertEqual(prompt.count(BOS), 1)
                self.assertEqual(parse_prompt(prompt), (role, report_prefix(role)))
                self.assertNotIn('<|start_header_id|>', prompt)
                self.assertNotIn('"$defs"', prompt)
                self.assertNotIn('"output_schema"', prompt)
                self.assertIn(TASKS[source].positive_definition, prompt)
                self.assertIn('"body": "UNIT"', prompt)

    def test_real_decoder_accepts_valid_reports_for_all_roles(self):
        for role in ROLES:
            report = fixture(role)
            validate_report(report, role, {'body': 'UNIT'})
            text = json.dumps(report, separators=(',', ':'))
            self.assertTrue(accepts(role, text), role)
            parser = seeded_parser(role, report_prefix(role))
            for char in text[len(report_prefix(role)):]:
                self.assertIn(char, parser.get_allowed_characters())
                parser = parser.add_character(char)
            self.assertTrue(parser.can_end())

    def test_real_decoder_rejects_structural_failure_modes(self):
        good = fixture('context')
        invalid = []
        bad = copy.deepcopy(good); bad['wrong_key'] = bad.pop('claimed_identity'); invalid.append(bad)
        bad = copy.deepcopy(good); bad.pop('uncertainties'); invalid.append(bad)
        bad = copy.deepcopy(good); bad['claimed_identity'] = []; invalid.append(bad)
        bad = copy.deepcopy(good); bad['claimed_identity']['status'] = 'certain'; invalid.append(bad)
        bad = copy.deepcopy(good); bad['role'] = 'action'; invalid.append(bad)
        for value in invalid:
            self.assertFalse(accepts('context', json.dumps(value, separators=(',', ':'))))
        self.assertFalse(accepts('context', '{}'))
        self.assertFalse(accepts('context', 'Here is the JSON'))
        self.assertFalse(accepts('context', '```json\n' + json.dumps(good)))

    def test_action_enum_is_enforced_inside_nested_record(self):
        report = fixture('action')
        record = shape('action')['requested_actions'][0]
        for name, value in [('action', 'UNIT'), ('statement_status', 'current_request')]:
            record[name] = dict(value=value, status='observed', scope='current',
                                evidence=[dict(field='body', quote='UNIT')])
        report['requested_actions'] = [record]
        report['no_actions_reason'] = dict(value=None, status='not_present', scope='unknown', evidence=[])
        self.assertTrue(accepts('action', json.dumps(report, separators=(',', ':'))))
        validate_report(report, 'action', {'body':'UNIT'})
        record['statement_status']['value'] = 'not_an_allowed_status'
        self.assertFalse(accepts('action', json.dumps(report, separators=(',', ':'))))

    def test_wrong_checkpoint_is_rejected_before_loading_weights(self):
        from unittest.mock import patch
        from four_text_v4.scoring.candidate_backend import HFScorer
        with patch('four_text_v4.scoring.candidate_backend.model_manifest', return_value={'config.json':'wrong-checkpoint'}):
            with self.assertRaisesRegex(ValueError, 'same frozen Llama-3.1-8B Base'):
                HFScorer(ROOT)

    def test_cross_field_and_source_errors_are_not_passed_as_correct(self):
        report = fixture('context')
        report['claimed_identity']['value'] = 'text despite unknown status'
        with self.assertRaisesRegex(ValueError, 'schema'):
            validate_report(report, 'context', {'body': 'UNIT'})
        report = fixture('context')
        report['communication_context']['evidence'][0]['quote'] = 'NOT IN INPUT'
        with self.assertRaisesRegex(ValueError, 'quotation'):
            validate_report(report, 'context', {'body': 'UNIT'})

    def test_unicode_quote_is_allowed_and_roundtrips(self):
        report = fixture('context')
        report['communication_context']['value'] = '邮件'
        report['communication_context']['evidence'][0]['quote'] = '邮件'
        text = json.dumps(report, ensure_ascii=False, separators=(',', ':'))
        self.assertTrue(accepts('context', text, '邮件'))
        validate_report(strict_json(text), 'context', {'body':'邮件'})

    def test_native_likelihood_scoring_stays_two_order_and_separate(self):
        class Backend:
            def __init__(self): self.calls = []
            def score(self, prompt, candidates):
                self.calls.append((prompt, candidates))
                return dict(logprobs=[-2., -1.], prompt_tokens=10, candidate_token_counts=[2,3])
        backend = Backend()
        result = score_task(backend, prompt_for('context'),
                            dict(text=json.dumps(fixture('context')), finish_reason='stop'))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['margin'], 1.)
        self.assertEqual(len(backend.calls), 2)
        for prompt, candidates in backend.calls:
            self.assertEqual(candidates, ('Safe Email', 'Phishing Email'))
            self.assertTrue(prompt.startswith(BOS))
            self.assertNotIn('REPORT_JSON:', prompt)
            self.assertNotIn('FIELDS FOR THIS ROLE', prompt)

    def test_truncation_never_reaches_candidate_scoring(self):
        class Backend:
            def score(self, *args): raise AssertionError('truncated report reached scoring')
        result = score_task(Backend(), prompt_for('context'),
                            dict(text=report_prefix('context'), finish_reason='length'))
        self.assertEqual(result['status'], 'review_required')
        self.assertEqual(result['reason'], 'ValueError: generation_length_limit')


if __name__ == '__main__':
    unittest.main()
