"""Report availability only: no lexical extraction, semantic scores or truth confidence."""
import math
from ..contracts.reports import all_claims

QUALITY_NAMES = ('has_current_claim', 'has_inferred_claim', 'has_noncurrent_claim',
                 'subject_present', 'sender_field_present', 'has_citation')
QUALITY_SIZE = len(QUALITY_NAMES)


def validate_quality(values, size=QUALITY_SIZE):
    if len(values) != size or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('complete bounded report-availability vector required')


def quality(mail, bundle):
    claims = [c for attempt in bundle.attempts for c in all_claims(attempt.report) if c['status'] in ('observed', 'inferred')]
    return (float(any(c['scope'] in ('current', 'mixed') for c in claims)),
            float(any(c['status'] == 'inferred' for c in claims)),
            float(any(c['scope'] in ('quoted', 'signature', 'mixed') for c in claims)),
            float(bool(mail.subject)), float(bool(mail.sender)), float(any(c['evidence'] for c in claims)))
