import re
import unicodedata
from email import policy
from email.parser import Parser
from ..contracts import RawEmail, CanonicalEmail
from ..util import digest
from .html_normalizer import visible_html, normalize_space
from .block_builder import blocks

VERSION='canonical-visible-email-v4.0'

def parse_email(raw: RawEmail) -> CanonicalEmail:
    if type(raw) is not RawEmail: raise TypeError('parse_email accepts RawEmail exactly, never prepared input')
    body,html,subject,sender=raw.body,raw.html,raw.subject,raw.sender
    missing=['sender_not_independently_verified']; attachments=[]
    if raw.mime:
        msg=Parser(policy=policy.default).parsestr(raw.mime)
        subject,sender=str(msg.get('subject','')),str(msg.get('from',''))
        plain=[]; rich=[]
        for part in msg.walk():
            if part.is_multipart():continue
            if part.get_content_disposition()=='attachment':
                attachments.append(part.get_filename() or 'unnamed');continue
            if part.get_content_type() not in ('text/plain','text/html'):continue
            try: text=part.get_content()
            except (LookupError,UnicodeError):
                text=(part.get_payload(decode=True) or b'').decode('utf8',errors='replace')
                missing.append('mime_decode_fallback')
            (plain if part.get_content_type()=='text/plain' else rich).append(str(text))
        body,html='\n\n'.join(plain),'\n\n'.join(rich)
    body=normalize_space(body); html_body,links=visible_html(html)
    if not body:body=html_body
    elif html_body:
        # Whitespace-equivalent alternatives contribute links, not a second body.
        key=lambda s:re.sub(r'\s+',' ',s).strip()
        if key(body)!=key(html_body):
            body+='\n\n[HTML alternative]\n'+html_body
            missing.append('differing_html_alternative_retained')
    if not sender:missing.append('sender_missing')
    if attachments:missing.append('attachment_contents_not_analyzed')
    fatal=None
    if not body or not any(c.isalpha() for c in body):fatal='no_readable_body'
    elif body.count('\ufffd')/len(body)>.2:fatal='decode_quality_insufficient'
    elif len(body)>500 and re.fullmatch(r'[A-Za-z0-9+/=\s]+',body) and max(map(len,body.split()),default=0)>200:fatal='probable_encoded_payload'
    if '\ufffd' in body:missing.append('replacement_characters')
    parts=blocks(body)
    fingerprint=digest(' '.join(unicodedata.normalize('NFKC',body).casefold().split()))
    return CanonicalEmail(raw,subject,sender,body,parts,links,tuple(attachments),tuple(dict.fromkeys(missing)),fatal,fingerprint,VERSION)
