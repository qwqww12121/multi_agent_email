"""Classification diagnostic over all frozen mails; evidence runs independently.

No calibration, PPO, threshold selection or evidence-based sample exclusion.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import importlib.metadata
from dataclasses import asdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_frozen_sample import SOURCES,code_manifest,append,write_new,summarize
from four_text_v5.contracts import RawEmail
from four_text_v5.scoring.candidate_backend import HFScorer
from four_text_v5.scoring.role_scorer import RoleScorer
from four_text_v5.pipeline import Pipeline
from four_text_v5.util import digest,read_jsonl,save_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--sample-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',required=True);args=p.parse_args()
    source=json.loads((args.sample_run/'data_manifest.json').read_text())
    source_runtime=json.loads((args.sample_run/'runtime.json').read_text())
    protocol=dict(version='v5-051-raw500-classification-20261030',calibration=False,ppo=False,rag=False,
        threshold=.5,normalization='mean_token',max_tokens=18432,budget_mode='reject',
        diagnostic_fusion='uncalibrated equal mean; not trained PPO',
        evidence_execution='independent 8-mail joint probe; not rerun for every classification',
        auxiliary_run=str(args.sample_run),auxiliary_run_signature=source_runtime['run_signature'],
        membership='exact same frozen 4x500 inputs, no evidence-dependent filtering')
    code=code_manifest()|{'scripts/'+Path(__file__).name:digest(Path(__file__).read_bytes())}
    if not args.out.exists():
        for s in SOURCES:
            for name,key in [('inputs.jsonl','inputs_sha256'),('labels.jsonl','labels_sha256')]:
                path=args.sample_run/'data'/s/name
                if digest(path.read_bytes())!=source['sources'][s][key]:raise ValueError('frozen sample hash mismatch')
                write_new(args.out/'data'/s/name,read_jsonl(path))
        save_json(args.out/'data_manifest.json',dict(protocol=protocol,code=code,sources=source['sources'],source_manifest=digest(source)))
    manifest=json.loads((args.out/'data_manifest.json').read_text())
    if manifest['code']!=code or manifest['protocol']!=protocol:raise ValueError('frozen classification contract changed')
    queries={}
    for s in SOURCES:
        path=args.out/'data'/s/'inputs.jsonl'
        if digest(path.read_bytes())!=manifest['sources'][s]['inputs_sha256']:raise ValueError('classification input changed')
        queries[s]=[RawEmail.from_dict(r) for r in read_jsonl(path)]
    import torch
    torch.set_num_threads(4)
    save_json(args.out/'progress.json',dict(phase='loading_model',pid=os.getpid(),time=time.time()))
    backend=HFScorer(args.model,max_tokens=18432,dtype='bfloat16',device='cuda:0');scorer=RoleScorer(backend)
    if scorer.signature!=source_runtime['scoring_signature']:raise ValueError('probe/full classification signatures differ')
    pipe=Pipeline(scorer,evidence=None)
    runtime=dict(protocol=protocol,scoring_signature=scorer.signature,
        run_signature=digest(dict(code=code,protocol=protocol,scoring=scorer.signature)),
        backend=backend.identity,pid=os.getpid(),gpu=torch.cuda.get_device_name(),
        versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers')},
        source_manifest=digest(manifest),expected_mails=2000)
    if (args.out/'runtime.json').exists():
        if json.loads((args.out/'runtime.json').read_text())['run_signature']!=runtime['run_signature']:raise ValueError('resume signature mismatch')
    else:save_json(args.out/'runtime.json',runtime)
    done={}
    if (args.out/'scores.jsonl').exists():
        for r in read_jsonl(args.out/'scores.jsonl'):
            if r['record_sha256']!=digest({k:v for k,v in r.items() if k!='record_sha256'}) or r['run_signature']!=runtime['run_signature']:raise ValueError('existing record hash/signature mismatch')
            key=(r['source'],r['id'])
            if key in done:raise ValueError('duplicate classification result')
            done[key]=r
    for i,raw in enumerate(queries[s][j] for j in range(500) for s in SOURCES):
        if (raw.source,raw.id) in done:continue
        save_json(args.out/'progress.json',dict(phase='classification',id=raw.id,source=raw.source,index=i+1,target=2000,pid=os.getpid(),time=time.time()))
        start=time.monotonic();result=pipe.run(raw,score_only=True);result.pop('record_sha256')
        result.update(seconds=time.monotonic()-start,run_signature=runtime['run_signature'])
        result['raw_equal_mean']=sum(r['normalized_label_score'] for r in result['roles'])/4 if result['status']=='scored' and len(result['roles'])==4 else None
        result['raw_equal_mean_note']='uncalibrated diagnostic; not PPO; auxiliary evidence has independent run'
        append(args.out/'scores.jsonl',result)
        print(json.dumps(dict(index=i+1,source=raw.source,id=raw.id,status=result['status'],seconds=result['seconds'],raw_equal_mean=result['raw_equal_mean'])),flush=True)
        if (i+1)%20==0:
            report=summarize(args.out);report.update(auxiliary_run=str(args.sample_run),evidence_execution='separate_run')
            save_json(args.out/'summary.json',report)
    report=summarize(args.out);report.update(auxiliary_run=str(args.sample_run),evidence_execution='separate_run')
    save_json(args.out/'summary.json',report)
    save_json(args.out/'progress.json',dict(phase='completed',target=2000,pid=os.getpid(),time=time.time(),run_signature=runtime['run_signature']))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
