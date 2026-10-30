"""Recompute first-stage transport and exact quote checks from saved raw outputs."""
import os,json,hashlib
from pathlib import Path
from collections import defaultdict
root = Path(os.environ.get('MAIL_AGENT_RUN_DIR', str(Path(__file__).resolve().parent))).resolve()
mails={x['id']:x for x in map(json.loads,(root/'inputs.jsonl').read_text(encoding='utf-8').splitlines())}
rows=list(map(json.loads,(root/'records.jsonl').read_text(encoding='utf-8').splitlines()))
assert len(rows)==64 and len({(r['arm'],r['role'],r['id']) for r in rows})==64
summary=defaultdict(lambda:dict(n=0,json_parseable=0,schema_valid=0,nonempty_exact_quote=0,length_limit=0,complete_object=0))
for r in rows:
 s=summary[r['arm']];s['n']+=1
 try:x=json.loads(r['raw']);s['json_parseable']+=1
 except ValueError:x=None
 valid=isinstance(x,dict) and set(x)=={'evidence'} and isinstance(x['evidence'],str) and len(x['evidence'])<=160
 exact=bool(valid and x['evidence'] and x['evidence'] in mails[r['id']]['body'])
 assert valid==r['valid_schema'] and exact==r['verbatim']
 s['schema_valid']+=int(valid);s['nonempty_exact_quote']+=int(exact)
 s['length_limit']+=r['finish']=='length';s['complete_object']+=r['finish']=='object'
short=list(map(json.loads,(root/'short_followup/records.jsonl').read_text(encoding='utf-8').splitlines()))
assert len(short)==32 and len({(r['style'],r['role'],r['id']) for r in short})==32
short_summary={style:{'n':sum(r['style']==style for r in short),'length_limit':sum(r['style']==style and r['finish']=='length' for r in short),'non_length_stop':sum(r['style']==style and r['finish']!='length' for r in short)} for style in ['qa','cloze']}
assert json.loads((root/'runtime.json').read_text())['model']['files']==json.loads((root/'short_followup/runtime.json').read_text())['model']['files']
report={'scope':'diagnostic only; exact quote and non-length stop are NOT semantic correctness','arms':dict(summary),'short_followup':short_summary,'files':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name in ['probe.py','short_probe.py','inputs.jsonl','config.json','runtime.json','records.jsonl']}}
(root/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report['arms'],indent=2))
