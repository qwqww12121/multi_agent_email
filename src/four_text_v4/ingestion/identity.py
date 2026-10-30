import re
from email import policy
from email.parser import Parser
from .html_normalizer import visible_html
from ..training.split_manifest import ExclusionRow,normalized_hash

def known_family(identifier):
    for pattern,name in ((r'(?:nazario|nazario_2024):(?:[^:]+:)?(\d+)','nazario-2024-mbox'),(r'(?:ceas08|ceas_2008_curated):(\d+)','ceas08-csv')):
        match=re.fullmatch(pattern,identifier)
        if match:return f'{name}:{int(match.group(1))}'
    return ''

def exclusion(mail):
    raw=mail.raw
    views={normalized_hash(v) for v in (raw.body,visible_html(raw.html)[0],mail.body) if v.strip()}
    if raw.mime:
        msg=Parser(policy=policy.default).parsestr(raw.mime)
        for part in msg.walk():
            if part.is_multipart() or part.get_content_disposition()=='attachment' or part.get_content_type() not in ('text/plain','text/html'):continue
            try:text=str(part.get_content())
            except (LookupError,UnicodeError):text=(part.get_payload(decode=True) or b'').decode('utf8',errors='replace')
            if part.get_content_type()=='text/html':text=visible_html(text)[0]
            if text.strip():views.add(normalized_hash(text))
    known=known_family(raw.id)
    if known and raw.family_id and known!=raw.family_id:raise ValueError('declared family contradicts known source identity')
    return ExclusionRow(raw.id,raw.source,raw.source_dataset,raw.task_type,raw.calibration_domain,
        mail.body,mail.body_hash,known or raw.family_id,tuple(sorted(views)),raw.input_protocol)
