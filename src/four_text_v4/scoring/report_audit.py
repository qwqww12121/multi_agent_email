"""Replay structural checks on saved reports without regenerating model content."""
from dataclasses import asdict
from ..contracts import ROLES
from ..contracts.reports import ReportAttempt, RoleReports, strict_json, validate_report, REPORT_VERSION
from ..ingestion.input_budget import slice_input
from ..experts.quality import quality
from ..util import digest, sigmoid
from .render import render_report, render_score, source_fields


def audit_saved_reports(record, mail):
    coverage = record['coverage']
    chunks = coverage['chunks']
    if coverage.get('complete') is not True or not chunks:
        raise ValueError('complete chunk coverage required')
    reconstructed = []
    end = 0
    for i, saved in enumerate(chunks):
        if saved['body_start'] != end or not end < saved['body_end'] <= len(mail.body):
            raise ValueError('chunk coverage gap or overlap')
        chunk = slice_input(mail, end, saved['body_end'], i, len(chunks))
        if digest(asdict(chunk)) != digest(saved):
            raise ValueError('canonical chunk/input mismatch')
        reconstructed.append(chunk); end = chunk.body_end
    if end != len(mail.body):
        raise ValueError('partial input coverage')
    if set(record['role_reports']) != set(ROLES):
        raise ValueError('four generated reports required')
    bundles = {}
    for score in record['roles']:
        role = score['role']; saved = record['role_reports'][role]
        if saved['role'] != role or saved['schema_version'] != REPORT_VERSION or len(saved['attempts']) != len(chunks):
            raise ValueError('report schema/role/chunk mismatch')
        attempts = []
        for chunk, a in zip(reconstructed, saved['attempts']):
            attempt = ReportAttempt(**a)
            expected = render_report(mail, chunk, role)
            if (attempt.error is not None or attempt.finish_reason != 'stop' or attempt.role != role
                    or attempt.chunk != chunk.index or attempt.prompt != expected or attempt.prompt_hash != digest(expected)):
                raise ValueError('generation provenance mismatch')
            parsed = validate_report(strict_json(attempt.raw_output), role, source_fields(mail, chunk))
            if parsed != attempt.report:
                raise ValueError('raw generation/parsed report mismatch')
            attempts.append(attempt)
        bundle = RoleReports(role, tuple(attempts)); bundles[role] = bundle
        if tuple(score['evidence_quality']) != quality(mail, bundle):
            raise ValueError('report availability mismatch')
        observations = score['observations']
        if len(observations) != 2 * len(chunks):
            raise ValueError('two native-score orders per report required')
        margin = sensitivity = 0.0
        for i, (chunk, attempt) in enumerate(zip(reconstructed, attempts)):
            pair = []
            for j, order in enumerate(('negative_first', 'positive_first')):
                obs = observations[2*i+j]
                prompt = render_score(mail, chunk, role, attempt.report, order)
                if obs['chunk'] != i or obs['definition_order'] != order or obs['prompt'] != prompt or obs['prompt_hash'] != digest(prompt):
                    raise ValueError('score/report prompt binding mismatch')
                pair.append(obs['positive_ll']-obs['negative_ll'])
            margin += sum(pair)/2*chunk.coverage
            sensitivity += abs(pair[0]-pair[1])*chunk.coverage
        if abs(score['margin']-margin)>1e-10 or abs(score['normalized_label_score']-sigmoid(margin))>1e-10 or abs(score['order_sensitivity']-sensitivity)>1e-10:
            raise ValueError('score aggregation mismatch')
    return bundles
