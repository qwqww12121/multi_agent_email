"""Model-authored role reports. Validation never extracts or repairs semantics."""
import json
from dataclasses import dataclass
from .task import ROLES

REPORT_VERSION = 'model-role-fields-4.0'
STATUSES = ('observed', 'inferred', 'not_present', 'unknown')
SCOPES = ('current', 'quoted', 'signature', 'mixed', 'unknown')

ACTION_FIELDS = ('requester', 'actor', 'action', 'action_object',
                 'destination_or_recipient', 'conditions', 'statement_status')
INFLUENCE_FIELDS = ('expression', 'mechanism', 'linked_action', 'promised_outcome',
                    'stated_consequence', 'authority_or_basis', 'communication_restrictions')
CONTEXT_FIELDS = ('claimed_identity', 'claimed_relationship', 'communication_context',
                  'prior_context', 'request_context_connection', 'inconsistencies_or_changes',
                  'missing_context', 'sender_domain', 'reply_to_domain')
PURPOSE_FIELDS = ('main_topic', 'stated_purpose', 'intended_outcome', 'direct_consequence',
                  'affected_objects_or_assets', 'beneficiary_or_receiver', 'subject_body_alignment')
LIST_FIELDS = {'action': ('visible_urls', 'attachment_mentions'),
               'context': ('brand_mentions', 'header_names'),
               'persuasion': (), 'purpose': ('subject_terms',)}
SINGLE_FIELDS = {'action': ('no_actions_reason',), 'context': CONTEXT_FIELDS,
                 'persuasion': ('no_influence_reason',), 'purpose': PURPOSE_FIELDS}


def unknown_claim():
    return dict(value=None, status='unknown', scope='unknown', evidence=[])


def shape(role):
    """A complete shape, not a generated report or inference fallback."""
    if role not in ROLES:
        raise ValueError('unknown role')
    result = dict(schema_version=REPORT_VERSION, role=role)
    result.update({key: unknown_claim() for key in SINGLE_FIELDS[role]})
    result.update({key: [] for key in LIST_FIELDS[role]})
    if role == 'action':
        result['requested_actions'] = [{key: unknown_claim() for key in ACTION_FIELDS}]
    if role == 'persuasion':
        result['influence_attempts'] = [{key: unknown_claim() for key in INFLUENCE_FIELDS}]
    result['uncertainties'] = unknown_claim()
    return result


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def _keys(obj, expected, path):
    if not isinstance(obj, dict) or set(obj) != set(expected):
        raise ValueError(f'{path}: exact fields required: {", ".join(expected)}')


def validate_report(report, role, sources):
    from .decoder_schema import validate_full_schema
    validate_full_schema(report, role)
    expected = shape(role)
    _keys(report, expected, 'report')
    if report['schema_version'] != REPORT_VERSION or report['role'] != role:
        raise ValueError('report role/version mismatch')

    def claim(item, path):
        _keys(item, ('value', 'status', 'scope', 'evidence'), path)
        value, status = item['value'], item['status']
        if status not in STATUSES or item['scope'] not in SCOPES:
            raise ValueError(path + ': invalid status/scope')
        if status in ('observed', 'inferred'):
            if not isinstance(value, str) or not value.strip() or len(value) > 1600:
                raise ValueError(path + ': supported values must be bounded nonempty text')
        elif value is not None:
            raise ValueError(path + ': absent/unknown value must be null')
        citations = item['evidence']
        if not isinstance(citations, list) or len(citations) > 12:
            raise ValueError(path + ': evidence must be a bounded list')
        if status in ('observed', 'inferred') and not citations:
            raise ValueError(path + ': supporting quotation required')
        for citation in citations:
            _keys(citation, ('field', 'quote'), path + '.evidence')
            field, quote = citation['field'], citation['quote']
            if (not isinstance(field, str) or field not in sources or not isinstance(quote, str)
                    or not quote.strip() or len(quote) > 2000 or quote not in sources[field]):
                raise ValueError(path + ': quotation not found in the visible input field')

    for key in (*SINGLE_FIELDS[role], 'uncertainties'):
        claim(report[key], key)
    for key in LIST_FIELDS[role]:
        items = report[key]
        if not isinstance(items, list) or len(items) > 64:
            raise ValueError(key + ': bounded list required')
        for i, item in enumerate(items):
            claim(item, f'{key}[{i}]')
    for key, fields, empty_key in (
            ('requested_actions', ACTION_FIELDS, 'no_actions_reason'),
            ('influence_attempts', INFLUENCE_FIELDS, 'no_influence_reason')):
        if key not in report:
            continue
        records = report[key]
        if not isinstance(records, list) or len(records) > 32:
            raise ValueError(key + ': bounded list required')
        for i, record in enumerate(records):
            _keys(record, fields, key)
            for field in fields:
                claim(record[field], f'{key}[{i}].{field}')
            anchor = record['action' if role == 'action' else 'expression']
            if anchor['status'] not in ('observed', 'inferred'):
                raise ValueError(key + ': a record must have a supported anchor')
            if role == 'action' and record['statement_status']['value'] not in (
                    'current_request', 'historical_description', 'quoted_request',
                    'conditional_request', 'negated_request', 'ambiguous'):
                raise ValueError('statement_status: controlled value required')
            if role == 'persuasion' and record['mechanism']['value'] not in (
                    'time_limit', 'promised_benefit', 'stated_loss', 'authority_or_rule',
                    'curiosity', 'secrecy_or_process_restriction', 'ordinary_reason', 'other', 'ambiguous'):
                raise ValueError('mechanism: controlled value required')
        if not records and report[empty_key]['status'] not in ('observed', 'inferred'):
            raise ValueError(empty_key + ': explain empty records with visible evidence')
        if records and report[empty_key]['status'] != 'not_present':
            raise ValueError(empty_key + ': must be not_present when records exist')
    return report


@dataclass(frozen=True)
class ReportAttempt:
    role: str
    chunk: int
    prompt: str
    prompt_hash: str
    raw_output: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str
    report: dict | None
    error: str | None


@dataclass(frozen=True)
class RoleReports:
    role: str
    attempts: tuple[ReportAttempt, ...]
    schema_version: str = REPORT_VERSION


def all_claims(value):
    if isinstance(value, dict):
        if set(value) == {'value', 'status', 'scope', 'evidence'}:
            yield value
        else:
            for child in value.values():
                yield from all_claims(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_claims(child)
