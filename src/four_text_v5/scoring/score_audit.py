"""Replay saved prompt bindings and arithmetic before joining training labels."""
import math
from dataclasses import asdict
from ..contracts import ROLES,TASKS
from ..ingestion.input_budget import slice_input
from ..util import digest,sigmoid
from .role_scorer import ORDERS,ScoringConfig
from .render import render_classification

def audit_saved_scores(record,mail):
    config=ScoringConfig(**record['scoring_config'])
    saved=record['coverage']['chunks']
    if not record['coverage']['complete'] or not saved:raise ValueError('complete input required')
    chunks=[];position=0
    for i,c in enumerate(saved):
        if c['body_start']!=position or c['body_end']<=position:raise ValueError('chunk coverage gap')
        chunk=slice_input(mail,position,c['body_end'],i,len(saved))
        if digest(asdict(chunk))!=digest(c):raise ValueError('saved chunk mismatch')
        chunks.append(chunk);position=c['body_end']
    if position!=len(mail.body):raise ValueError('incomplete body')
    if tuple(s['role'] for s in record['roles'])!=ROLES:raise ValueError('role order mismatch')
    close=lambda a,b:math.isfinite(a) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
    for s in record['roles']:
        if s['input_signature']!=digest(asdict(mail)) or tuple(s['candidates'])!=TASKS[mail.raw.source].candidates:raise ValueError('input/candidate binding mismatch')
        if len(s['observations'])!=len(chunks)*len(ORDERS):raise ValueError('observations missing')
        margin=0.;sensitivity=0.
        for chunk in chunks:
            pair=[]
            for j,order in enumerate(ORDERS):
                o=s['observations'][chunk.index*2+j];prompt=render_classification(mail,chunk,s['role'],order)
                if o['chunk']!=chunk.index or o['definition_order']!=order or o['prompt']!=prompt or o['prompt_hash']!=digest(prompt):raise ValueError('prompt provenance mismatch')
                if o['normalization']!=config.normalization:raise ValueError('normalization mismatch')
                counts=o['candidate_tokens'];sums=o['summed_logprobs']
                if len(counts)!=2 or len(sums)!=2 or any(type(v) is not int or v<1 for v in counts) or any(not math.isfinite(v) or v>0 for v in sums):raise ValueError('invalid saved likelihoods')
                ll=[v/n if config.normalization=='mean_token' else v for v,n in zip(sums,counts)]
                if not close(o['negative_ll'],ll[0]) or not close(o['positive_ll'],ll[1]):raise ValueError('candidate arithmetic mismatch')
                pair.append(ll[1]-ll[0])
            margin+=sum(pair)/2*chunk.coverage;sensitivity+=abs(pair[0]-pair[1])*chunk.coverage
        if not close(s['margin'],margin) or not close(s['normalized_label_score'],sigmoid(margin)) or not close(s['order_sensitivity'],sensitivity):raise ValueError('role aggregation mismatch')
    return True
