"""Retrospective frozen-sample diagnostic; no calibration, PPO or output repair.

Classification scores are committed before running auxiliary evidence. Per-call
raw evidence and progress survive an interrupted mail. No label enters prompts.
"""
import argparse
import collections
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from four_text_v5.contracts import RawEmail,ROLES,TASKS
from four_text_v5.contracts.domain import INPUT_PROTOCOL
from four_text_v5.contracts.evidence import strict_json,validate
from four_text_v5.ingestion import parse_email,InputBudget
from four_text_v5.scoring.candidate_backend import HFScorer
from four_text_v5.scoring.role_scorer import RoleScorer
from four_text_v5.scoring.evidence import EvidenceAnalyzer,EvidenceConfig
from four_text_v5.pipeline import Pipeline
from four_text_v5.util import digest,read_jsonl,save_json

SOURCES=('phishingpot','enron_spam','ephishllm','kaggle')
PROTOCOL=dict(version='v5.0.5.1-frozen-diagnostic-20261030',per_source=500,sample_seed=20261029,
    membership='exact frozen R3 IDs/order; verified input and label file hashes',
    calibration=False,ppo=False,rag=False,evidence=True,normalization='mean_token',
    max_tokens=18432,budget_mode='reject',evidence_max_new_tokens=1024,evidence_retries=1,
    quote_mode='exact',threshold=.5,diagnostic_fusion='uncalibrated equal mean; not PPO prediction',
    interpretation='retrospective previously examined test pool, not a new blind heldout')

def code_manifest():
    paths=list((ROOT/'src/four_text_v5').rglob('*.py'))+list((ROOT/'src/four_text_v5/prompts').rglob('*.json'))+list((ROOT/'src/four_text_v5/prompts').rglob('*.txt'))+[Path(__file__)]
    return {p.relative_to(ROOT).as_posix():digest(p.read_bytes()) for p in sorted(paths)}

def append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    row=dict(row,record_sha256=digest(row))
    with path.open('a',encoding='utf8') as stream:
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())

def write_new(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def prepare(previous,out):
    if out.exists():raise ValueError('new run directory required')
    old=json.loads((previous/'data_manifest.json').read_text())
    sources={}
    for source in SOURCES:
        m=old['sources'][source];inputs=previous/'data'/source/'inputs.jsonl';labels=previous/'data'/source/'labels.jsonl'
        if digest(inputs.read_bytes())!=m['input_sha256'] or digest(labels.read_bytes())!=m['labels_sha256']:raise ValueError('previous frozen file hash mismatch')
        original=read_jsonl(inputs);truth=read_jsonl(labels)
        if len(original)!=500 or [r['id'] for r in original]!=m['ids']:raise ValueError('frozen sample membership mismatch')
        if [(r['source_dataset'],r['id']) for r in original]!=[(r['source_dataset'],r['id']) for r in truth]:raise ValueError('frozen label identities mismatch')
        converted=[]
        for row in original:
            adapted=dict(row,input_protocol=INPUT_PROTOCOL,calibration_domain=source+'-raw-diagnostic-v5',recipient=row.get('recipient',''))
            # Mail content and source membership remain byte-for-byte equal as strings.
            assert all(adapted[k]==row[k] for k in row if k not in ('input_protocol','calibration_domain'))
            RawEmail.from_dict(adapted);converted.append(adapted)
        write_new(out/'data'/source/'inputs.jsonl',converted)
        write_new(out/'data'/source/'labels.jsonl',truth)
        sources[source]=dict(selected_n=500,ids=m['ids'],class_counts=m['class_counts'],
            old_inputs_sha256=m['input_sha256'],old_labels_sha256=m['labels_sha256'],
            inputs_sha256=digest((out/'data'/source/'inputs.jsonl').read_bytes()),
            labels_sha256=digest((out/'data'/source/'labels.jsonl').read_bytes()))
    save_json(out/'data_manifest.json',dict(previous=str(previous),previous_manifest_sha256=digest((previous/'data_manifest.json').read_bytes()),
        protocol=PROTOCOL,sources=sources,code=code_manifest(),adaptation='only input protocol, diagnostic domain and explicit empty recipient metadata'))
    print(json.dumps(dict(prepared=str(out),sources={s:x['selected_n'] for s,x in sources.items()})),flush=True)

def summarize(out):
    manifest=json.loads((out/'data_manifest.json').read_text())
    scores=read_jsonl(out/'scores.jsonl') if (out/'scores.jsonl').exists() else []
    evidence=read_jsonl(out/'evidence.jsonl') if (out/'evidence.jsonl').exists() else []
    report=dict(classification_mails=len(scores),classified_all_four=sum(len(r.get('roles',[]))==4 and r['status']=='scored' for r in scores),
        evidence_mails=len(evidence),sources={},evidence_role_records=0,evidence_roles_valid=0,evidence_roles_without_valid_nonempty_quote=0,
        evidence_features_valid=0,evidence_features_unreliable=0,evidence_features_found=0,evidence_all_four_valid_mails=0,
        evidence_failures={},calibration=False,ppo=False)
    errors=collections.Counter()
    for source in SOURCES:
        labels=read_jsonl(out/'data'/source/'labels.jsonl');truth={r['id']:r['label'] for r in labels}
        done=[r for r in scores if r['source']==source];cells=collections.Counter();role_cells={role:collections.Counter() for role in ROLES}
        for row in done:
            y=truth[row['id']]
            for s in row.get('roles',[]):
                p=int(s['normalized_label_score']>=.5);role_cells[s['role']].update(dict(n=1,correct=int(p==y),positive_predictions=p))
            if row.get('raw_equal_mean') is None:cells['unknown']+=1;continue
            p=int(row['raw_equal_mean']>=.5);cells['valid']+=1;cells['correct']+=int(p==y)
            cells['tp' if y and p else 'fn' if y else 'fp' if p else 'tn']+=1
        report['sources'][source]=dict(expected=500,scored=len(done),raw_equal_mean=dict(cells),
            accuracy_processed=cells['correct']/len(done) if done else None,accuracy_all_500=cells['correct']/500,
            roles={r:dict(c) for r,c in role_cells.items()})
    for row in evidence:
        records=row['evidence'].get('records',[])
        report['evidence_all_four_valid_mails']+=int(len(records)==4 and all(r['status']=='valid' for r in records))
        for r in records:
            report['evidence_role_records']+=1;report['evidence_roles_valid']+=int(r['status']=='valid')
            found=0
            for f in r['features']:
                good=f['status']=='valid';report['evidence_features_valid']+=int(good);report['evidence_features_unreliable']+=int(not good)
                if good and f['result']['found']:found+=1;report['evidence_features_found']+=1
                if not good:errors[f.get('error','unavailable')]+=1
            report['evidence_roles_without_valid_nonempty_quote']+=int(found==0)
        if not records:errors[row['evidence'].get('error','unavailable_evidence')]+=1
    report['evidence_failures']=dict(errors)
    report['limitation']='Small prefix accuracy is diagnostic only; schema/quote validity does not establish semantic correctness.'
    save_json(out/'summary.json',report);return report

class AuditedBackend:
    def __init__(self,backend,out):self.backend,self.out=backend,out;self.context={};self.calls=0
    def __getattr__(self,name):return getattr(self.backend,name)
    def score(self,*args):return self.backend.score(*args)
    def generate_evidence(self,prompt,budget,contract,prefix):
        role=contract['properties']['agent_role']['enum'][0];features=contract['properties']['features']['items']['properties']['feature_id']['enum']
        self.calls+=1;start=time.monotonic()
        event=dict(self.context,phase='evidence_generation',role=role,features=features,call=self.calls)
        save_json(self.out/'progress.json',dict(event,pid=os.getpid(),time=time.time()))
        print(json.dumps(dict(event,event='start')),flush=True)
        try:
            answer=self.backend.generate_evidence(prompt,budget,contract,prefix)
            append(self.out/'generation_attempts.jsonl',dict(event,prompt=prompt,prompt_hash=digest(prompt),schema=contract,output=answer,seconds=time.monotonic()-start))
            print(json.dumps(dict(event,event='finish',finish_reason=answer['finish_reason'],generated_tokens=answer.get('generated_tokens'),seconds=time.monotonic()-start)),flush=True)
            return answer
        except Exception as exc:
            append(self.out/'generation_attempts.jsonl',dict(event,prompt=prompt,prompt_hash=digest(prompt),error=f'{type(exc).__name__}: {exc}',seconds=time.monotonic()-start));raise

def run(args):
    out=args.out;manifest=json.loads((out/'data_manifest.json').read_text())
    if manifest['protocol']!=PROTOCOL or manifest['code']!=code_manifest():raise ValueError('frozen protocol/source changed')
    queries={}
    for source in SOURCES:
        path=out/'data'/source/'inputs.jsonl'
        if digest(path.read_bytes())!=manifest['sources'][source]['inputs_sha256']:raise ValueError('frozen input changed')
        queries[source]=[RawEmail.from_dict(r) for r in read_jsonl(path)]
    import torch
    torch.set_num_threads(4)
    save_json(out/'progress.json',dict(phase='loading_model',pid=os.getpid(),time=time.time()))
    backend=AuditedBackend(HFScorer(args.model,max_tokens=18432,dtype='bfloat16',device='cuda:0'),out)
    scorer=RoleScorer(backend);analyzer=EvidenceAnalyzer(backend);pipe=Pipeline(scorer,evidence=None)
    runtime=dict(protocol=PROTOCOL,scoring_signature=scorer.signature,evidence_signature=analyzer.signature,
        run_signature=digest(dict(code=manifest['code'],protocol=PROTOCOL,scoring=scorer.signature,evidence=analyzer.signature)),
        backend=backend.identity,versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','lm-format-enforcer')},
        pid=os.getpid(),gpu=torch.cuda.get_device_name(),limit=args.limit,source_manifest=digest(manifest))
    if (out/'runtime.json').exists():
        old=json.loads((out/'runtime.json').read_text())
        if old['run_signature']!=runtime['run_signature']:raise ValueError('cannot resume different signature')
    else:save_json(out/'runtime.json',runtime)
    print(json.dumps(dict(model_ready=True,**runtime)),flush=True)
    completed_scores={(r['source'],r['id']):r for r in read_jsonl(out/'scores.jsonl')} if (out/'scores.jsonl').exists() else {}
    completed_evidence={(r['source'],r['id']) for r in read_jsonl(out/'evidence.jsonl')} if (out/'evidence.jsonl').exists() else set()
    target=[queries[s][i] for i in range(500) for s in SOURCES][:args.limit]
    for index,raw in enumerate(target):
        backend.context=dict(id=raw.id,source=raw.source,index=index+1,target=args.limit,run_signature=runtime['run_signature'])
        key=(raw.source,raw.id)
        if key not in completed_scores:
            save_json(out/'progress.json',dict(backend.context,phase='classification',pid=os.getpid(),time=time.time()))
            start=time.monotonic();result=pipe.run(raw,score_only=True)
            result.pop('record_sha256');result['seconds']=time.monotonic()-start
            result['raw_equal_mean']=sum(r['normalized_label_score'] for r in result['roles'])/4 if result['status']=='scored' and len(result['roles'])==4 else None
            result['raw_equal_mean_note']='uncalibrated equal mean diagnostic; not trained PPO';result['run_signature']=runtime['run_signature']
            append(out/'scores.jsonl',result);completed_scores[key]=result
            print(json.dumps(dict(backend.context,classification_status=result['status'],classification_seconds=result['seconds'],raw_equal_mean=result['raw_equal_mean'])),flush=True)
        if key not in completed_evidence:
            start=time.monotonic()
            try:result=analyzer.analyze(parse_email(raw))
            except Exception as exc:result=dict(signature=analyzer.signature,status='unreliable',error=f'{type(exc).__name__}: {exc}')
            append(out/'evidence.jsonl',dict(backend.context,evidence=result,seconds=time.monotonic()-start));completed_evidence.add(key)
        summarize(out)
    summary=summarize(out)
    save_json(out/'progress.json',dict(phase='completed_requested_prefix',target=args.limit,pid=os.getpid(),time=time.time(),run_signature=runtime['run_signature']))
    print(json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','summarize']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--model');p.add_argument('--limit',type=int,default=8);args=p.parse_args()
    if args.command=='prepare':prepare(args.previous,args.out)
    elif args.command=='summarize':print(json.dumps(summarize(args.out)),flush=True)
    else:
        if not 1<=args.limit<=2000:raise ValueError('limit out of range')
        run(args)

if __name__=='__main__':main()
