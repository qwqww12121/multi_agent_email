"""Greedy Base completion with per-row schema token masks; no repair/fallback."""
import importlib.metadata
from ..util import digest

DECODER_VERSION = '0.11.3'


def object_end(text):
    start = len(text) - len(text.lstrip())
    if start == len(text) or text[start] != '{':
        return None
    depth = 0
    quoted = escaped = False
    for i in range(start, len(text)):
        c = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def seeded_parser(schema, prefix, alphabet_text=''):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    from lmformatenforcer.consts import COMPLETE_ALPHABET
    if importlib.metadata.version('lm-format-enforcer') != DECODER_VERSION:
        raise RuntimeError('Install pinned lm-format-enforcer; no free-generation fallback')
    config = CharacterLevelParserConfig(
        alphabet=''.join(sorted(set(COMPLETE_ALPHABET + alphabet_text))),
        force_json_field_order=True, max_consecutive_whitespaces=12,
        max_json_array_length=64)
    parser = JsonSchemaParser(schema, config=config)
    for char in prefix:
        if char not in parser.get_allowed_characters():
            raise ValueError('structural prefix incompatible with decoder schema')
        parser = parser.add_character(char)
    return parser


def generate_json(backend, prompt, max_new_tokens, schema, prefix):
    prompts=[prompt]
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn)
    tokenizer = backend.tokenizer
    specs = [(schema, prefix)]
    rows = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    if not rows or any(not row or len(row) + max_new_tokens > backend.max_tokens for row in rows):
        raise ValueError('full input plus output allowance exceeds context budget')
    if any(row[0] != tokenizer.bos_token_id for row in rows):
        raise ValueError('missing Base BOS token')
    width = max(map(len, rows)); eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError('explicit Base EOS required')
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    device = next(backend.model.parameters()).device
    ids = torch.tensor([[pad] * (width-len(row)) + row for row in rows], device=device)
    mask = torch.tensor([[0] * (width-len(row)) + [1] * len(row) for row in rows], device=device)
    if not hasattr(backend, '_r4_tokenizer_data'):
        backend._r4_tokenizer_data = build_token_enforcer_tokenizer_data(
            tokenizer, vocab_size=backend.model.config.vocab_size)
    callbacks = []
    for (role, prefix), prompt in zip(specs, prompts):
        parser = seeded_parser(role, prefix, prompt)
        settings = parser.config
        callback = build_transformers_prefix_allowed_tokens_fn(backend._r4_tokenizer_data, parser)
        # LMFE 0.11.3 replaces the root config when building the token enforcer.
        # Restore the explicitly signed settings instead of environment defaults.
        callback.token_enforcer.root_parser.config = settings
        callbacks.append(callback)

    def allowed(batch_id, sent):
        choices = callbacks[batch_id](batch_id, sent)
        if not choices:
            raise RuntimeError('decoder has no valid continuation')
        state = callbacks[batch_id].token_enforcer.prefix_states[tuple(sent.tolist())]
        if eos in choices and not state.parser.can_end():
            raise RuntimeError('decoder failed before completing the schema')
        return choices

    completed_lengths = {}
    class CompleteObject(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            finished = []
            for i, ((_, prefix), row) in enumerate(zip(specs, input_ids)):
                done = object_end(prefix + tokenizer.decode(row[width:].tolist(), skip_special_tokens=True)) is not None
                if done:
                    completed_lengths.setdefault(i, len(row) - width)
                finished.append(done)
            return torch.tensor(finished, device=input_ids.device)

    with torch.inference_mode():
        generated = backend.model.generate(input_ids=ids, attention_mask=mask,
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
            eos_token_id=eos, pad_token_id=pad, prefix_allowed_tokens_fn=allowed,
            stopping_criteria=StoppingCriteriaList([CompleteObject()]))
    results = []
    for i, ((role, prefix), row, seq) in enumerate(zip(specs, rows, generated[:, width:].tolist())):
        if i in completed_lengths:
            seq = seq[:completed_lengths[i]]  # exclude batch padding after this row stopped
        ended = eos in seq
        if ended:
            seq = seq[:seq.index(eos) + 1]
        suffix = tokenizer.decode(seq, skip_special_tokens=True)
        combined = prefix + suffix
        end = object_end(combined)
        results.append(dict(text=combined[:end] if end is not None else combined,
            raw_text=suffix, prefill_text=prefix,
            trailing_text=combined[end:] if end is not None else '',
            prompt_tokens=len(row), generated_tokens=len(seq),
            finish_reason='stop' if end is not None or ended else 'length',
            stop_cause='complete_json_object' if end is not None else 'eos' if ended else 'length',
            decoder='lm-format-enforcer', decoder_version=DECODER_VERSION,
            decoder_schema_hash=digest(role),
            constraint_scope='structure/types/keys/enums; full schema, citations and meaning checked separately'))
    return results[0]
