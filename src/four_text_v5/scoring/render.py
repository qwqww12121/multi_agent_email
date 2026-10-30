"""Two independent prompts sharing one role catalog and raw email input."""
import json
from string import Template
from email import policy
from email.parser import Parser
from ..contracts import TASKS
from ..experts.catalog import load_role, render_catalog, resource

BOS = '<|begin_of_text|>'
VERSION = 'role-driven-dual-prompt-5.0'


def source_fields(mail, chunk):
    headers = ''
    if mail.raw.mime:
        message = Parser(policy=policy.default).parsestr(mail.raw.mime, headersonly=True)
        headers = '\n'.join(f'{k}: {v}' for k,v in message.raw_items())
    return dict(subject=mail.subject, sender=mail.sender, recipient=mail.recipient,
                body=chunk.text, headers=headers,
                html_links=json.dumps(mail.links,ensure_ascii=False),
                attachment_metadata=json.dumps(mail.attachments,ensure_ascii=False))


def task_meaning(mail):
    if mail.raw.task_type == 'spam':
        return ('SPAM task: identify unsolicited or unwanted correspondence, including unwanted bulk promotion and scams. '
                'Commercial content alone does not establish missing consent. Legitimate wanted correspondence is negative.')
    return ('Phishing task: identify deceptive attempts to obtain sensitive information, funds or access, deliver malicious content, '
            'or induce harmful actions. Ordinary advertising alone is insufficient. The task is not generic spam detection.')


def email_input(mail,chunk):
    return json.dumps(dict(scope=dict(index=chunk.index,total=chunk.total,body_start=chunk.body_start,
        body_end=chunk.body_end,unavailable=list(mail.missing),unseen_relations='unknown'),
        fields=source_fields(mail,chunk)),ensure_ascii=False)


def render_classification(mail,chunk,role,order='negative_first'):
    task=TASKS[mail.raw.source];config=load_role(role)
    definitions=[(task.negative,task.negative_definition),(task.positive,task.positive_definition)]
    if order=='positive_first':definitions.reverse()
    elif order!='negative_first':raise ValueError('invalid definition order')
    text=Template(resource('common/classification_base.txt')).substitute(
        task=task_meaning(mail),definitions=json.dumps(definitions,ensure_ascii=False),
        role_definition=config['role_definition'],feature_catalog=render_catalog(role),
        exclusion_rules='\n'.join('- '+s for s in config['exclusion_rules']),
        email=email_input(mail,chunk),labels=json.dumps([d[0] for d in definitions],ensure_ascii=False))
    return BOS+text.rstrip()


def render_evidence(mail,chunk,role,selected=None,retry=False):
    config=load_role(role)
    text=Template(resource('common/evidence_base.txt')).substitute(
        task=task_meaning(mail),role_definition=config['role_definition'],
        feature_catalog=render_catalog(role,selected),role=role,
        exclusion_rules='\n'.join('- '+s for s in config['exclusion_rules']),
        retry_instruction=('This is a bounded retry for only the assigned failed features. Reassess the original email; do not reproduce an earlier answer.' if retry else ''),
        email=email_input(mail,chunk))
    prefix='{"agent_role":'+json.dumps(role)+',"features":'
    return BOS+text+prefix,prefix
