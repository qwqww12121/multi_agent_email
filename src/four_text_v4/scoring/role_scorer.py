import importlib.metadata
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from ..contracts import ROLES, TASKS, RawRoleScore, CandidateObservation
from ..contracts.reports import ReportAttempt, RoleReports, strict_json, validate_report, REPORT_VERSION
from ..ingestion import InputBudget, plan_inputs
from ..util import digest, sigmoid
from .render import render_report, render_score, source_fields

ORDERS = ('negative_first', 'positive_first')


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 4096
    do_sample: bool = False

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 64 or self.do_sample is not False:
            raise ValueError('positive bounded report budget and deterministic generation required')


class RoleScoringFailure(RuntimeError):
    def __init__(self, role, chunk, phase, error, completed, partial, attempts=()):
        super().__init__(f'{role}/chunk={chunk}/{phase}: {type(error).__name__}: {error}')
        self.completed, self.partial, self.attempts = completed, partial, attempts


class RoleScorer:
    def __init__(self, backend, budget=InputBudget(), generation=GenerationConfig()):
        if not callable(getattr(backend, 'generate_report', None)):
            raise TypeError('v4 requires model generation; rule-extraction/score-only backends are unsupported')
        self.backend, self.budget, self.generation = backend, budget, generation
        root = Path(__file__).resolve().parents[1]
        versions = {}
        for name in ('torch', 'transformers', 'tokenizers', 'lm-format-enforcer', 'jsonschema', 'pydantic', 'interegular'):
            try: versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: versions[name] = None
        self.signature = digest(dict(backend=backend.identity, budget=asdict(budget),
            generation=asdict(generation), report_schema=REPORT_VERSION, orders=ORDERS,
            aggregation='same-report-two-order-margin; character-weighted-complete-chunks',
            failure_policy='single-generation-no-repair-no-fallback',
            code={p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in sorted(root.rglob('*.py'))},
            versions=versions))

    def _tokens(self, text):
        return len(self.backend.tokenizer.encode(text, add_special_tokens=False))

    def score(self, mail):
        if mail.fatal: raise ValueError('unusable canonical input: ' + mail.fatal)
        from ..experts.quality import quality
        candidates = TASKS[mail.raw.source].candidates
        limit = min(self.budget.max_tokens, getattr(self.backend, 'max_tokens', self.budget.max_tokens))
        def fits(chunk):
            return all(self._tokens(render_report(mail, chunk, role)) + self.generation.max_new_tokens <= limit
                and all(self._tokens(render_score(mail, chunk, role, {}, order))
                        + self.generation.max_new_tokens + max(self._tokens(' ' + c) for c in candidates) <= limit
                        for order in ORDERS) for role in ROLES)
        chunks = plan_inputs(mail, self.budget, fits)
        scores, bundles, all_attempts = [], {}, []
        for role in ROLES:
            observations, margins, sensitivity, attempts = [], [], [], []
            for chunk in chunks:
                prompt = render_report(mail, chunk, role)
                raw_output, prompt_tokens, generated_tokens, finish_reason = '', self._tokens(prompt), 0, 'error'
                try:
                    answer = self.backend.generate_report(prompt, self.generation.max_new_tokens)
                    raw_output = answer['text']
                    prompt_tokens, generated_tokens = answer['prompt_tokens'], answer['generated_tokens']
                    finish_reason = answer['finish_reason']
                    if finish_reason != 'stop': raise ValueError('report generation did not finish before the token limit')
                    report = validate_report(strict_json(raw_output), role, source_fields(mail, chunk))
                except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                    attempt = ReportAttempt(role, chunk.index, prompt, digest(prompt), raw_output,
                        prompt_tokens, generated_tokens, finish_reason, None, str(exc))
                    raise RoleScoringFailure(role, chunk.index, 'report_generation', exc,
                        tuple(scores), tuple(observations), tuple(all_attempts + [attempt])) from exc
                attempt = ReportAttempt(role, chunk.index, prompt, digest(prompt), raw_output,
                    prompt_tokens, generated_tokens, finish_reason, report, None)
                attempts.append(attempt); all_attempts.append(attempt)
                pair = []
                for order in ORDERS:
                    prompt = render_score(mail, chunk, role, report, order)
                    try:
                        if self._tokens(prompt) + max(self._tokens(' ' + c) for c in candidates) > limit:
                            raise ValueError('complete report plus email exceeds scoring budget; no truncation')
                        answer = self.backend.score(prompt, candidates)
                        ll = answer['logprobs']
                        if len(ll) != 2 or not all(math.isfinite(v) for v in ll): raise ValueError('invalid likelihoods')
                    except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                        raise RoleScoringFailure(role, chunk.index, order, exc,
                            tuple(scores), tuple(observations), tuple(all_attempts)) from exc
                    observation = CandidateObservation(chunk.index, order, prompt, digest(prompt), *ll,
                        answer['prompt_tokens'], tuple(answer['candidate_token_counts']))
                    observations.append(observation); pair.append(observation.margin)
                margins.append(sum(pair) / 2 * chunk.coverage)
                sensitivity.append(abs(pair[0] - pair[1]) * chunk.coverage)
            bundle = RoleReports(role, tuple(attempts)); bundles[role] = bundle
            margin = sum(margins)
            scores.append(RawRoleScore(role, mail.raw.source, self.signature, candidates, tuple(observations),
                margin, sigmoid(margin), sum(sensitivity), min(1.0, math.fsum(c.coverage for c in chunks)),
                mail.raw.source_dataset, mail.raw.task_type, mail.raw.calibration_domain, mail.raw.input_protocol,
                digest(dict(id=mail.raw.id, source_dataset=mail.raw.source_dataset, body_hash=mail.body_hash)), quality(mail, bundle)))
        return tuple(scores), bundles, chunks
