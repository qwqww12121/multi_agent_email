"""v4r4 Base completion prompts and constrained named-object reports; no calibration/PPO.

Roles continue independently. Invalid reports are retained without repair.
"""
import argparse
import collections
from dataclasses import asdict
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from four_text_v4.contracts import ROLES,TASKS,RawEmail
from four_text_v4.contracts.domain import task_for
from four_text_v4.contracts.reports import strict_json,validate_report,all_claims
from four_text_v4.experts.cards import FIXED_EXAMPLE_MAILS,PROMPT_VERSION
from four_text_v4.ingestion import parse_email,InputBudget,plan_inputs,BudgetExceeded
from four_text_v4.scoring.render import render_report,render_score,source_fields
from four_text_v4.scoring.candidate_backend import HFScorer
from four_text_v4.scoring.json_transport import generate_json_batch
from four_text_v4.scoring.role_scorer import RoleScorer,GenerationConfig,ORDERS
from four_text_v4.training.split_manifest import normalized_hash
from four_text_v4.util import digest,sigmoid,read_jsonl,save_json

SOURCES=('phishingpot','enron_spam','ephishllm','kaggle')
EXPECTED={'phishingpot':3856,'enron_spam':6497,'ephishllm':3324,'kaggle':3600}
PROTOCOL=dict(name='model-fields-v4r4-base-constrained-raw500-20261029',seed=20261029,per_source=500,
    prompt_version=PROMPT_VERSION,prompt_demonstrations=0,
    report_encoding='named claim objects; model-role-fields-4.0',
    schema_enforcement='lm-format-enforcer==0.11.3 token mask; conditional and evidence checks after generation',
    prompt_transport='Base BOS plus plain completion; no chat template',
    prefill='structural schema_version/role only; no semantic values',
    decoder_field_order='fixed; main relation first; 12 consecutive whitespace cap',
    sampling='sort sha256(JSON seed/source/id), first 500; independent of labels',
    threshold=.5,aggregation='equal mean of all four successful uncalibrated sigmoid margins',
    calibration=False,ppo=False,rag=False,max_tokens=18432,report_max_new_tokens=1536,
    budget_mode='reject',role_failure='independent; no replacement, repair or retry',
    max_generation_batch=8,batch_token_budget=30000,termination='first complete JSON object or EOS',
    batching='fixed maximum eight; numerical differences recorded, not selected by test performance',
    interpretation='retrospective historical test pool; not a new blind holdout')


def freeze_code():
    return {p.relative_to(ROOT).as_posix():digest(p.read_bytes())
            for p in sorted((ROOT/'src/four_text_v4').rglob('*.py'))} | {
                'scripts/'+Path(__file__).name:digest(Path(__file__).read_bytes())}


def adapt(row,source):
    return RawEmail(id=row['id'],label_schema=source,source_dataset=row['dataset'],
        task_type=task_for(source),calibration_domain=source+'-raw-diagnostic-v4',
        subject=row.get('subject','') or '',sender=row.get('sender','') or '',
        body=row.get('body_text','') or '',html=row.get('body_html','') or '')


def write_new(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def prepare(base,out):
    if (out/'data_manifest.json').exists():raise ValueError('manifest exists; preparation is frozen')
    excluded={normalized_hash(t) for t in FIXED_EXAMPLE_MAILS}; manifest={}
    for source in SOURCES:
        path=base/'native_generation_20261014_v2/tests'/source/'test.jsonl'
        rows=read_jsonl(path)
        if len(rows)!=EXPECTED[source] or len({r['id'] for r in rows})!=len(rows):raise ValueError('source membership/count mismatch')
        ordered=sorted(rows,key=lambda r:digest(dict(seed=PROTOCOL['seed'],source=source,id=r['id'])))
        chosen=[];exclusions=[]
        for row in ordered:
            raw=adapt(row,source);mail=parse_email(raw)
            if mail.body_hash in excluded:
                exclusions.append(dict(id=raw.id,reason='fixed_prompt_example'));continue
            chosen.append((row,raw,mail))
            if len(chosen)==PROTOCOL['per_source']:break
        if len(chosen)!=500:raise ValueError('insufficient source samples')
        queries=[asdict(raw) for _,raw,_ in chosen]
        labels=[dict(id=raw.id,source_dataset=raw.source_dataset,label=int(row['label'])) for row,raw,_ in chosen]
        if any(r['label'] not in (0,1) for r in labels):raise ValueError('invalid label')
        write_new(out/'data'/source/'inputs.jsonl',queries)
        write_new(out/'data'/source/'labels.jsonl',labels)
        manifest[source]=dict(source_file=str(path),source_file_sha256=digest(path.read_bytes()),
            source_n=len(rows),selected_n=len(chosen),class_counts=dict(collections.Counter(r['label'] for r in labels)),
            ids=[r['id'] for r in queries],input_sha256=digest((out/'data'/source/'inputs.jsonl').read_bytes()),
            labels_sha256=digest((out/'data'/source/'labels.jsonl').read_bytes()),exclusions=exclusions,
            unique_canonical_bodies=len({m.body_hash for _,_,m in chosen}))
    save_json(out/'data_manifest.json',dict(protocol=PROTOCOL,code=freeze_code(),sources=manifest))
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


def reuse_sample(previous,out):
    if out.exists():raise ValueError('new run output must not exist')
    old=json.loads((previous/'data_manifest.json').read_text())
    for source in SOURCES:
        for name,key in (('inputs.jsonl','input_sha256'),('labels.jsonl','labels_sha256')):
            path=previous/'data'/source/name
            if digest(path.read_bytes())!=old['sources'][source][key]:raise ValueError('prior sample hash mismatch')
    out.mkdir(parents=True)
    shutil.copytree(previous/'data',out/'data')
    # The zero-shot version has no example texts; preserve the generic split guard.
    example_hashes={normalized_hash(t) for t in FIXED_EXAMPLE_MAILS}
    for source in SOURCES:
        if any(parse_email(RawEmail.from_dict(r)).body_hash in example_hashes
               for r in read_jsonl(out/'data'/source/'inputs.jsonl')):raise ValueError('authored demonstration collides with frozen inputs')
    save_json(out/'data_manifest.json',dict(protocol=PROTOCOL,code=freeze_code(),sources=old['sources'],
        parent_manifest_sha256=digest(old),parent_run=str(previous),sample_reused_byte_for_byte=True))


def generate_batch(backend,prompts,max_new_tokens):
    return generate_json_batch(backend,prompts,max_new_tokens)


def accept_execution(backend,out):
    # Independent synthetic transport probes, never test-label-driven selection.
    from four_text_v4.ingestion.input_budget import slice_input
    raw = RawEmail(id='transport-only', label_schema='kaggle', source_dataset='kaggle',
                   task_type='phishing', calibration_domain='transport-only',
                   body='Transport verification input.')
    mail = parse_email(raw)
    chunk = slice_input(mail, 0, len(mail.body))
    prompts = [render_report(mail, chunk, role) for role in ROLES]
    start=time.monotonic()
    sequential=[backend.generate_report(p,64) for p in prompts]
    batch=generate_batch(backend,prompts,64)
    equal=all(a['text']==b['text'] and a['generated_tokens']==b['generated_tokens'] for a,b in zip(sequential,batch))
    selected=PROTOCOL['max_generation_batch']
    result=dict(selected_batch=selected,exact_greedy_prefix_match=equal,prefix_limit=64,
        elapsed_seconds=time.monotonic()-start,sequential=sequential,batched=batch,
        scope='synthetic diagnostic only; BF16 batch-dependent differences expected and retained; fixed max batch eight with token-budget packing')
    save_json(out/'execution_acceptance.json',result)
    return result


def score_task(backend,task,answer):
    raw,mail,chunk,role,prompt=task
    result=dict(id=raw.id,source=raw.source,role=role,status='review_required',probability=None,
        prediction=None,native_prediction=None,report_prompt=prompt,report_prompt_hash=digest(prompt),
        generation=answer,report=None,candidate_observations=[])
    try:
        if answer['finish_reason']!='stop':raise ValueError('generation_length_limit')
        report=validate_report(strict_json(answer['text']),role,source_fields(mail,chunk))
        result['report']=report
        claims=list(all_claims(report))
        result['claim_status_counts']=dict(collections.Counter(c['status'] for c in claims))
        result['supported_claims']=sum(c['status'] in ('observed','inferred') for c in claims)
        result['content_status']='supported_fields_present' if result['supported_claims'] else 'empty_analysis'
        pair=[]
        for order in ORDERS:
            scored_prompt=render_score(mail,chunk,role,report,order)
            score=backend.score(scored_prompt,TASKS[raw.source].candidates)
            if len(score['logprobs'])!=2 or not all(math.isfinite(x) for x in score['logprobs']):raise ValueError('invalid likelihoods')
            pair.append(score['logprobs'][1]-score['logprobs'][0])
            result['candidate_observations'].append(dict(order=order,prompt=scored_prompt,prompt_hash=digest(scored_prompt),
                candidates=TASKS[raw.source].candidates,**score))
        margin=sum(pair)/2;p=sigmoid(margin);pred=int(p>=.5)
        result.update(status='ok',margin=margin,probability=p,prediction=pred,
            native_prediction=TASKS[raw.source].candidates[pred],order_sensitivity=abs(pair[0]-pair[1]),reason=None)
    except (ValueError,RuntimeError,TypeError,KeyError) as exc:
        result['reason']=f'{type(exc).__name__}: {exc}'
    return result


def append_result(out,result,signature):
    result['run_signature']=signature;result['record_sha256']=digest(result)
    path=out/'results'/result['source']/(result['role']+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf8') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def existing(out):
    rows={}
    for path in (out/'results').glob('*/*.jsonl'):
        for row in read_jsonl(path):
            if row['record_sha256']!=digest({k:v for k,v in row.items() if k!='record_sha256'}):raise ValueError('record hash mismatch')
            key=(row['source'],row['id'],row['role'])
            if key in rows:raise ValueError('duplicate role result')
            rows[key]=row
    return rows


def metrics(labels,predictions):
    counts=collections.Counter();errors=[]
    for row in labels:
        y=row['label'];counts['n']+=1;counts['positive' if y else 'negative']+=1
        p=predictions.get(row['id'])
        if p is None:counts['pending']+=1;continue
        counts['processed']+=1
        if p.get('prediction') is None:counts['unknown']+=1;continue
        counts['valid']+=1;pred=p['prediction']
        counts['tp' if y and pred else 'fn' if y else 'fp' if pred else 'tn']+=1
        errors.append((p['probability']-y)**2)
    result={k:counts[k] for k in ('n','positive','negative','processed','pending','valid','unknown','tp','tn','fp','fn')}
    result.update(accuracy_all=(counts['tp']+counts['tn'])/counts['n'],
        accuracy_processed=(counts['tp']+counts['tn'])/counts['processed'] if counts['processed'] else None,
        accuracy_valid=(counts['tp']+counts['tn'])/counts['valid'] if counts['valid'] else None,
        recall_all_positive=counts['tp']/counts['positive'] if counts['positive'] else None,
        precision=counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,
        coverage=counts['valid']/counts['n'],brier_valid=sum(errors)/len(errors) if errors else None)
    return result


def summarize(out):
    rows=existing(out);summary={}
    for source in SOURCES:
        labels=read_jsonl(out/'data'/source/'labels.jsonl');ensemble={}
        roles={role:{i:r for (s,i,k),r in rows.items() if s==source and k==role} for role in ROLES}
        for row in labels:
            entries=[roles[r].get(row['id']) for r in ROLES]
            if any(r is None for r in entries):continue
            if any(r['status']!='ok' for r in entries):ensemble[row['id']]=dict(prediction=None,probability=None)
            else:
                p=sum(r['probability'] for r in entries)/4
                ensemble[row['id']]=dict(prediction=int(p>=.5),probability=p)
        summary[source]=dict(roles={r:metrics(labels,roles[r]) for r in ROLES},equal_mean=metrics(labels,ensemble),
            empty_supported_reports={role:sum(r.get('supported_claims')==0 for r in roles[role].values()) for role in ROLES},
            failures=dict(collections.Counter(r.get('reason') for (s,_,_),r in rows.items() if s==source and r['status']!='ok')))
    payload=dict(protocol=PROTOCOL,role_records=len(rows),expected_role_records=8000,complete=len(rows)==8000,sources=summary)
    save_json(out/'summary.json',payload)
    return payload


def run(args):
    out=args.out;manifest=json.loads((out/'data_manifest.json').read_text())
    if manifest['protocol']!=PROTOCOL or manifest['code']!=freeze_code():raise ValueError('frozen protocol/code changed')
    queries={}
    for source in SOURCES:
        path=out/'data'/source/'inputs.jsonl'
        if digest(path.read_bytes())!=manifest['sources'][source]['input_sha256']:raise ValueError('frozen input changed')
        if digest((out/'data'/source/'labels.jsonl').read_bytes())!=manifest['sources'][source]['labels_sha256']:raise ValueError('frozen labels changed')
        queries[source]=[RawEmail.from_dict(r) for r in read_jsonl(path)]
    import torch
    torch.set_num_threads(4)
    backend=HFScorer(args.model,max_tokens=PROTOCOL['max_tokens'],dtype='bfloat16',device='cuda:0')
    scorer=RoleScorer(backend,InputBudget(PROTOCOL['max_tokens']),GenerationConfig(PROTOCOL['report_max_new_tokens']))
    acceptance_path=out/'execution_acceptance.json'
    acceptance=json.loads(acceptance_path.read_text()) if acceptance_path.exists() else accept_execution(backend,out)
    signature=digest(dict(scoring=scorer.signature,manifest=digest(manifest),execution=acceptance,protocol=PROTOCOL))
    identity=dict(run_signature=signature,model=backend.identity,scoring_signature=scorer.signature,execution=acceptance)
    if (out/'runtime.json').exists() and json.loads((out/'runtime.json').read_text())!=identity:raise ValueError('resume runtime mismatch')
    save_json(out/'runtime.json',identity)
    previous=existing(out)
    if any(r['run_signature']!=signature for r in previous.values()):raise ValueError('mixed result signatures')
    batch=[];start=time.monotonic();new_count=0
    def progress():
        save_json(out/'progress.json',dict(status='running',completed_roles=len(previous)+new_count,total_roles=8000,
            elapsed_this_process=time.monotonic()-start,pid=os.getpid(),run_signature=signature))
    def flush():
        nonlocal new_count,batch
        if not batch:return
        progress();t=time.monotonic()
        try:
            answers=generate_batch(backend,[task[-1] for task in batch],PROTOCOL['report_max_new_tokens']) if len(batch)>1 else [backend.generate_report(batch[0][-1],PROTOCOL['report_max_new_tokens'])]
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():torch.cuda.empty_cache()
            for raw,mail,chunk,role,prompt in batch:
                append_result(out,dict(id=raw.id,source=raw.source,role=role,status='review_required',prediction=None,probability=None,
                    reason='generation_backend: '+str(exc),report_prompt=prompt,report=None),signature)
                new_count+=1
        else:
            for task,answer in zip(batch,answers):
                result=score_task(backend,task,answer);result['generation_batch_size']=len(batch)
                result['batch_generation_seconds']=time.monotonic()-t
                append_result(out,result,signature);new_count+=1
        batch=[];progress()
        print(json.dumps(dict(completed_roles=len(previous)+new_count,total_roles=8000,elapsed=time.monotonic()-start)),flush=True)
        if (len(previous)+new_count)%40==0:summarize(out)
    progress()
    for i in range(500):
        for source in SOURCES:
            raw=queries[source][i];mail=parse_email(raw)
            if args.limit_mails and i>=args.limit_mails:break
            def fits(chunk):
                return all(scorer._tokens(render_report(mail,chunk,r))+PROTOCOL['report_max_new_tokens']<=PROTOCOL['max_tokens']
                           and all(scorer._tokens(render_score(mail,chunk,r,{},o))+PROTOCOL['report_max_new_tokens']+
                                   max(scorer._tokens(' '+c) for c in TASKS[source].candidates)<=PROTOCOL['max_tokens'] for o in ORDERS) for r in ROLES)
            reason=mail.fatal
            try:chunks=() if reason else plan_inputs(mail,scorer.budget,fits)
            except BudgetExceeded as exc:reason='input_budget: '+str(exc);chunks=()
            for role in ROLES:
                if (source,raw.id,role) in previous:continue
                if reason:
                    append_result(out,dict(id=raw.id,source=source,role=role,status='review_required',prediction=None,probability=None,reason=reason,report=None),signature);new_count+=1;continue
                chunk=chunks[0];prompt=render_report(mail,chunk,role)
                cost=scorer._tokens(prompt)+PROTOCOL['report_max_new_tokens']
                if batch and (len(batch)>=acceptance['selected_batch'] or sum(scorer._tokens(t[-1])+PROTOCOL['report_max_new_tokens'] for t in batch)+cost>PROTOCOL['batch_token_budget']):flush()
                batch.append((raw,mail,chunk,role,prompt))
            if (out/'STOP_AFTER_BATCH').exists():
                flush();summary=summarize(out)
                save_json(out/'progress.json',dict(status='paused',completed_roles=summary['role_records'],total_roles=8000,
                    elapsed_this_process=time.monotonic()-start,pid=os.getpid(),run_signature=signature))
                return
        if args.limit_mails and i>=args.limit_mails:break
    flush();summary=summarize(out)
    save_json(out/'progress.json',dict(status='complete' if summary['complete'] else 'partial',completed_roles=summary['role_records'],total_roles=8000,
        elapsed_this_process=time.monotonic()-start,pid=os.getpid(),run_signature=signature))
    if summary['complete']:save_json(out/'complete.json',dict(run_signature=signature,summary_sha256=digest(summary),role_records=8000))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','reuse-sample','run','summarize'])
    p.add_argument('--previous',type=Path)
    p.add_argument('--base',type=Path,default=Path('/phishing_email/test_1'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',default='/phishing_email/test_1/Llama-3.1-8B');p.add_argument('--limit-mails',type=int,default=0)
    args=p.parse_args()
    if args.mode=='prepare':prepare(args.base,args.out)
    elif args.mode=='reuse-sample':reuse_sample(args.previous,args.out)
    elif args.mode=='summarize':print(json.dumps(summarize(args.out),ensure_ascii=False,indent=2))
    else:run(args)


if __name__=='__main__':main()
