"""One route: model-generated reports -> native likelihoods -> calibration -> PPO."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from .contracts import RawEmail,ROLES
from .contracts.domain import DomainPolicy
from .ingestion import parse_email,InputBudget
from .ingestion.identity import exclusion
from .scoring.candidate_backend import HFScorer
from .scoring.role_scorer import RoleScorer,GenerationConfig
from .pipeline import Pipeline
from .training.split_manifest import TrainingRow,ExclusionRow
from .training.train_fusion import train_bundle
from .fusion.ppo import PPOConfig
from .evaluation import evaluate
from .util import read_jsonl,digest


def write_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)


def join_labels(scores,labels):
    index={(r['source_dataset'],r['id']):r['label'] for r in labels}
    if len(index)!=len(labels) or any(type(y) is not int or y not in (0,1) for y in index.values()):raise ValueError('unique binary labels required')
    if {(r['source_dataset'],r['id']) for r in scores}!=set(index) or len(scores)!=len(index):raise ValueError('score/label source identity mismatch')
    rows=[];excluded=[]
    for r in scores:
        if r.get('record_sha256')!=digest({k:v for k,v in r.items() if k!='record_sha256'}):raise ValueError('score record hash mismatch')
        if r['status']!='scored':
            excluded.append(dict(source_dataset=r['source_dataset'],id=r['id'],reason=r['reason']));continue
        mail=parse_email(RawEmail.from_dict(r['input']['raw']));member=exclusion(mail)
        if r['body_hash']!=member.body_hash or r['input']['body']!=mail.body or r['family_id']!=member.family_id or set(r['content_hashes'])!=set(member.content_hashes):raise ValueError('canonical score input mismatch')
        bindings=dict(id=mail.raw.id,source=mail.raw.source,source_dataset=mail.raw.source_dataset,task_type=mail.raw.task_type,calibration_domain=mail.raw.calibration_domain,input_protocol=mail.raw.input_protocol)
        if any(r[k]!=v for k,v in bindings.items()):raise ValueError('score input domain mismatch')
        if tuple(s['role'] for s in r['roles'])!=ROLES:raise ValueError('role order mismatch')
        for s in r['roles']:
            if any(s[k]!=r[k] for k in ('source','source_dataset','task_type','calibration_domain','input_protocol','scoring_signature')) or abs(s['coverage']-1)>1e-8:raise ValueError('role binding/coverage mismatch')
            if s['input_signature']!=digest(dict(id=mail.raw.id,source_dataset=mail.raw.source_dataset,body_hash=mail.body_hash)):raise ValueError('role input signature mismatch')
        from .scoring.report_audit import audit_saved_reports
        audit_saved_reports(r,mail)
        rows.append(asdict(TrainingRow(**asdict(member),label=index[(r['source_dataset'],r['id'])],margins=tuple(s['margin'] for s in r['roles']),scoring_signature=r['scoring_signature'],evidence_quality=tuple(v for s in r['roles'] for v in s['evidence_quality']))))
    return rows,excluded


def main(argv=None):
    parser=argparse.ArgumentParser(description='Four-text v4: model-authored role reports, native scores and domain-bound PPO')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('score','predict'):
        p=sub.add_parser(name);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
        p.add_argument('--model',required=True);p.add_argument('--device',default='auto');p.add_argument('--dtype',default='bfloat16')
        p.add_argument('--max-tokens',type=int,default=18432);p.add_argument('--budget-mode',choices=['reject','chunk_all'],default='reject');p.add_argument('--max-chunks',type=int,default=16)
        p.add_argument('--report-max-new-tokens',type=int,default=4096)
        if name=='predict':p.add_argument('--bundle',required=True)
    p=sub.add_parser('manifest');p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('join-labels');p.add_argument('--scores',required=True);p.add_argument('--labels',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('train')
    for name in ('calibration','fusion','validation','heldout-manifest','domain-policy','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--ppo-config',help='Frozen PPOConfig JSON; costs and threshold chosen before fitting')
    p.add_argument('--threads',type=int,default=1)
    p=sub.add_parser('evaluate');p.add_argument('--predictions',required=True);p.add_argument('--labels',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if Path(args.output).exists():raise ValueError('output exists; use a new versioned path')
    load_json=lambda path:json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if args.command in ('score','predict'):
        backend=HFScorer(args.model,args.max_tokens,args.dtype,args.device)
        scorer=RoleScorer(backend,InputBudget(args.max_tokens,args.max_chunks,args.budget_mode),GenerationConfig(args.report_max_new_tokens))
        pipe=Pipeline(scorer,load_json(args.bundle) if args.command=='predict' else None)
        write_rows(args.output,(pipe.run(RawEmail.from_dict(r),score_only=args.command=='score') for r in read_jsonl(args.input)))
    elif args.command=='manifest':
        write_rows(args.output,(asdict(exclusion(parse_email(RawEmail.from_dict(r)))) for r in read_jsonl(args.input)))
    elif args.command=='join-labels':
        sidecar=str(args.output)+'.excluded.json'
        if Path(sidecar).exists():raise ValueError('exclusion report exists')
        rows,excluded=join_labels(read_jsonl(args.scores),read_jsonl(args.labels))
        write_rows(args.output,rows);write_json(sidecar,excluded)
    elif args.command=='train':
        import torch
        if args.threads<1:raise ValueError('positive thread count required')
        torch.set_num_threads(args.threads)
        load=lambda path:[TrainingRow(**r) for r in read_jsonl(path)]
        policy=DomainPolicy.from_dict(load_json(args.domain_policy))
        cfg=PPOConfig(**load_json(args.ppo_config)) if args.ppo_config else PPOConfig(threshold=policy.threshold)
        write_json(args.output,train_bundle(load(args.calibration),load(args.fusion),load(args.validation),[ExclusionRow(**r) for r in read_jsonl(args.heldout_manifest)],policy,cfg))
    else:write_json(args.output,evaluate(read_jsonl(args.predictions),read_jsonl(args.labels)))


if __name__=='__main__':main()
