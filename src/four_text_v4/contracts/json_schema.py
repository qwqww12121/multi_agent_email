"""Explicit named-object schemas, with no filled reports or example emails."""
from copy import deepcopy
from .reports import (REPORT_VERSION, STATUSES, SCOPES, SINGLE_FIELDS, LIST_FIELDS,
                      ACTION_FIELDS, INFLUENCE_FIELDS, shape)

DESCRIPTION = {
    'requester': 'Person or organization presented as making the request; not authenticated identity.',
    'actor': 'Person or group expected to carry out the action.',
    'action': 'Requested or described behavior; preserve negation and distinguish current, quoted and past actions.',
    'action_object': 'Object, information or asset on which the behavior operates.',
    'destination_or_recipient': 'Visible destination or recipient of the operation or its output.',
    'conditions': 'Stated timing, prerequisites or circumstances attached to the request.',
    'statement_status': 'Relationship of the action to the current message: active, historical, quoted, conditional, negated or ambiguous.',
    'no_actions_reason': 'Supported explanation of an empty requested_actions list. Cite a passage supporting the reading; do not invent a no-action quotation.',
    'visible_urls': 'URL strings actually visible in supplied input; do not infer hidden destinations.',
    'attachment_mentions': 'Attachments mentioned in text or supplied metadata; their unseen contents remain unknown.',
    'claimed_identity': 'Who the sender presents themselves as or represents; a brand mention alone is not affiliation.',
    'claimed_relationship': 'Claimed or supported inferred relationship between sender and recipient.',
    'communication_context': 'Visible matter or situation prompting the communication.',
    'prior_context': 'Earlier contact or agreements referenced or included; distinguish a claim from an included exchange.',
    'request_context_connection': 'How the request follows from the visible relationship and situation, or a supported gap in that connection.',
    'inconsistencies_or_changes': 'Specific visible contradictions or changes. Cite both competing passages; do not invent expected procedures.',
    'missing_context': 'Missing information needed to resolve a supported context question; missing consent does not establish lack of consent.',
    'sender_domain': 'Domain in the supplied sender or From address; unknown when unavailable. Do not substitute URL domains.',
    'reply_to_domain': 'Domain in an explicitly supplied Reply-To address; unknown when unavailable.',
    'brand_mentions': 'Brands or organizations actually mentioned, without authenticating affiliation.',
    'header_names': 'Names of email header fields actually supplied in headers, not personal display names.',
    'expression': 'Actual wording used to influence behavior; connect it to its intended action.',
    'mechanism': 'time_limit: time constraint; promised_benefit: gain; stated_loss: adverse outcome; authority_or_rule: authority or obligation; curiosity: withheld information; secrecy_or_process_restriction: limits on disclosure or verification; ordinary_reason: practical rationale; other: another supported mechanism; ambiguous: unresolved mechanism.',
    'linked_action': 'Behavior the expression is intended to encourage or discourage.',
    'promised_outcome': 'Benefit or favorable outcome the sender promises for compliance.',
    'stated_consequence': 'Consequence the sender claims will follow noncompliance.',
    'authority_or_basis': 'Person, organization, rule or rationale invoked as support; not verified authenticity.',
    'communication_restrictions': 'Stated restrictions on consultation, disclosure, verification or channels; do not infer restrictions from silence.',
    'no_influence_reason': 'Supported explanation of an empty influence_attempts list, with a real passage supporting that reading.',
    'main_topic': 'Main subject matter of the visible message.',
    'stated_purpose': 'Explicit reason the sender gives for contacting the recipient.',
    'intended_outcome': 'Response or state the message seeks to bring about; mark inferred if not explicit.',
    'direct_consequence': 'Immediate supported result of complying, keeping conditions explicit and avoiding speculative later harm.',
    'affected_objects_or_assets': 'Objects or assets directly involved; do not assume that involvement establishes risk.',
    'beneficiary_or_receiver': 'Who receives the requested item or directly benefits; real-world identity may remain unknown.',
    'subject_body_alignment': 'Relationship between supplied subject and body; cite both for a mismatch; unknown if subject is unavailable.',
    'subject_terms': 'Meaningful terms actually present in subject; do not invent a subject from body.',
    'uncertainties': 'Unresolved issues limiting this role. Use supported findings when quotable; otherwise unknown/null/empty evidence. Use not_present/null/empty evidence if no relevant uncertainty is identified.',
}

ACTION_STATUSES = ('current_request', 'historical_description', 'quoted_request',
                   'conditional_request', 'negated_request', 'ambiguous')
MECHANISMS = ('time_limit', 'promised_benefit', 'stated_loss', 'authority_or_rule',
              'curiosity', 'secrecy_or_process_restriction', 'ordinary_reason', 'other', 'ambiguous')


def object_schema(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def field_schema(name, supported=False, values=None):
    result = {'$ref': '#/$defs/claim', 'description': DESCRIPTION[name]}
    if supported or values:
        props = {'status': {'enum': ['observed', 'inferred']}}
        if values:
            props['value'] = {'enum': list(values)}
        result['allOf'] = [dict(properties=props)]
    return result


def report_schema(role):
    shape(role)  # Validate role only; no dummy report is inserted into the prompt.
    properties = dict(schema_version={'const': REPORT_VERSION}, role={'const': role})
    properties.update({name: field_schema(name) for name in (*SINGLE_FIELDS[role], 'uncertainties')})
    properties.update({name: dict(type='array', maxItems=64, items=field_schema(name),
                                 description=DESCRIPTION[name]) for name in LIST_FIELDS[role]})
    record_key = 'requested_actions' if role == 'action' else 'influence_attempts' if role == 'persuasion' else None
    if record_key:
        fields = ACTION_FIELDS if role == 'action' else INFLUENCE_FIELDS
        record = {}
        for name in fields:
            values = ACTION_STATUSES if name == 'statement_status' else MECHANISMS if name == 'mechanism' else None
            record[name] = field_schema(name, supported=name in ('action', 'expression'), values=values)
        properties[record_key] = dict(type='array', maxItems=32, items=object_schema(record))
    schema = object_schema(properties)
    schema['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
    citation = object_schema(dict(field=dict(type='string', enum=['subject','sender','body','headers','html_links','attachment_metadata']),
                                  quote=dict(type='string', minLength=1, maxLength=2000)))
    claim = object_schema(dict(value=dict(type=['string','null'], minLength=1, maxLength=1600),
                               status=dict(enum=list(STATUSES)), scope=dict(enum=list(SCOPES)),
                               evidence=dict(type='array', maxItems=12, items={'$ref':'#/$defs/citation'})))
    claim['allOf'] = [{
        'if': {'properties': {'status': {'enum': ['observed', 'inferred']}}},
        'then': {'properties': {'value': {'type':'string'}, 'evidence': {'minItems':1}}},
        'else': {'properties': {'value': {'type':'null'}}},
    }]
    schema['$defs'] = dict(claim=claim, citation=citation)
    if record_key:
        reason = 'no_actions_reason' if role == 'action' else 'no_influence_reason'
        schema['allOf'] = [{
            'if': {'properties': {record_key: {'maxItems':0}}},
            'then': {'properties': {reason: {'properties': {'status': {'enum':['observed','inferred']}}}}},
            'else': {'properties': {reason: {'properties': {'status': {'const':'not_present'}}}}},
        }]
    return deepcopy(schema)
