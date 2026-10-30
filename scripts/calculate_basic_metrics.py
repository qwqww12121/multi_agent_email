"""Offline metrics for frozen V5 diagnostic; never tunes thresholds or repairs outputs."""
import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from four_text_v5.contracts import ROLES,RawEmail
from four_text_v5.contracts.evidence import strict_json,validate
from four_text_v5.experts.catalog import feature_ids
from four_text_v5.ingestion import parse_email
from four_text_v5.ingestion.input_budget import slice_input
from four_text_v5.scoring.render import source_fields
from four_text_v5.util import digest

SOURCES=('phishingpot','enron_spam','ephishllm','kaggle')
PREFIX='raw500_classification_20261030'
PROBE='probe8_20261030'

def ratio(a,b):return a/b if b else None

def auc(p,y):
    positive=sum(y);negative=len(y)-positive
    if not positive or not negative:return None
    ordered=sorted(zip(p,y));rank_sum=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        rank_sum+=sum(label for _,label in ordered[i:j])*(i+1+j)/2
        i=j
    return (rank_sum-positive*(positive+1)/2)/(positive*negative)

def metrics(rows):
    valid=[r for r in rows if r['p'] is not None];cells=collections.Counter()
    for r in valid:
        y=r['y'];pred=r['p']>=.5
        cells['tp' if y and pred else 'fn' if y else 'fp' if pred else 'tn']+=1
    tp,tn,fp,fn=(cells[k] for k in ('tp','tn','fp','fn'))
    bins=[[] for _ in range(10)]
    for r in valid:bins[min(9,int(r['p']*10))].append(r)
    support_positive=sum(r['y'] for r in rows);support_negative=len(rows)-support_positive
    unknown_positive=sum(r['y']==1 and r['p'] is None for r in rows)
    unknown_negative=sum(r['y']==0 and r['p'] is None for r in rows)
    reliability=[dict(lower=i/10,upper=(i+1)/10,n=len(b),mean_probability=ratio(sum(r['p'] for r in b),len(b)),
        positive_fraction=ratio(sum(r['y'] for r in b),len(b))) for i,b in enumerate(bins)]
    return dict(n=len(rows),n_scored=len(valid),n_unknown=len(rows)-len(valid),coverage=ratio(len(valid),len(rows)),
        positive_support=support_positive,negative_support=support_negative,unknown_positive=unknown_positive,unknown_negative=unknown_negative,
        tp=tp,tn=tn,fp=fp,fn=fn,positive_predictions=tp+fp,
        accuracy_all=ratio(tp+tn,len(rows)),accuracy_scored=ratio(tp+tn,len(valid)),precision=ratio(tp,tp+fp),
        recall=ratio(tp,tp+fn),f1=ratio(2*tp,2*tp+fp+fn),fnr=ratio(fn,tp+fn),ham_recall=ratio(tn,tn+fp),fpr=ratio(fp,tn+fp),
        recall_all_positive=ratio(tp,support_positive),miss_or_abstain_all_positive=ratio(fn+unknown_positive,support_positive),
        ham_recall_all_negative=ratio(tn,support_negative),
        brier=ratio(sum((r['p']-r['y'])**2 for r in valid),len(valid)),
        ece10=ratio(sum(abs(sum(r['p']-r['y'] for r in b)) for b in bins),len(valid)),
        roc_auc=auc([r['p'] for r in valid],[r['y'] for r in valid]),reliability=reliability)

def correlation(a,b):
    n=len(a)
    numerator=n*sum(x*y for x,y in zip(a,b))-sum(a)*sum(b)
    denominator=math.sqrt((n*sum(x*x for x in a)-sum(a)**2)*(n*sum(x*x for x in b)-sum(b)**2))
    return numerator/denominator if denominator else None

def complementary(rows):
    result=[]
    for left,right in itertools.combinations(ROLES,2):
        cohort=[r for r in rows if r['roles'].get(left) is not None and r['roles'].get(right) is not None]
        a=[int((r['roles'][left]>=.5)!=r['y']) for r in cohort];b=[int((r['roles'][right]>=.5)!=r['y']) for r in cohort]
        both=sum(x*y for x,y in zip(a,b))
        result.append(dict(left=left,right=right,n=len(cohort),error_correlation=correlation(a,b),joint_error_count=both,
            joint_error_rate=ratio(both,len(cohort)),disagreement_rate=ratio(sum(x!=y for x,y in zip(a,b)),len(cohort))))
    complete=[r for r in rows if all(r['roles'].get(role) is not None for role in ROLES)]
    counts=collections.Counter(sum((r['roles'][role]>=.5)!=r['y'] for role in ROLES) for r in complete)
    return dict(pairs=result,complete_n=len(complete),wrong_agent_count_distribution={str(i):counts[i] for i in range(5)},
        all_four_wrong_count=counts[4],all_four_wrong_rate=ratio(counts[4],len(complete)))

def schema_ok(v,s):
    typ=s.get('type')
    if typ=='object':
        if type(v) is not dict or set(s.get('required',[]))-set(v):return False
        props=s.get('properties',{})
        if s.get('additionalProperties') is False and set(v)-set(props):return False
        if any(not schema_ok(x,props[k]) for k,x in v.items() if k in props):return False
    elif typ=='array':
        if type(v) is not list or not s.get('minItems',0)<=len(v)<=s.get('maxItems',float('inf')):return False
        if any(not schema_ok(x,s.get('items',{})) for x in v):return False
    elif typ=='string':
        if type(v) is not str or len(v)>s.get('maxLength',float('inf')):return False
    elif typ=='boolean':
        if type(v) is not bool:return False
    return 'enum' not in s or v in s['enum']

def evidence_metrics(tar):
    read=lambda name:[json.loads(s) for s in tar.extractfile(PROBE+'/'+name) if s.strip()]
    inputs={(s,r['id']):r for s in SOURCES for r in read('data/'+s+'/inputs.jsonl')}
    final=read('evidence.jsonl');attempts=read('generation_attempts.jsonl')
    final_records={(r['source'],r['id'],s['role']):s for r in final for s in r['evidence']['records']}
    attempt_metrics=[];emitted=collections.defaultdict(set);first_seen=set()
    for attempt in attempts:
        assert attempt['record_sha256']==digest({k:v for k,v in attempt.items() if k!='record_sha256'})
        key=(attempt['source'],attempt['id'],attempt['role']);first=key not in first_seen;first_seen.add(key)
        raw=RawEmail.from_dict(inputs[key[:2]]);mail=parse_email(raw);chunk=slice_input(mail,0,len(mail.body))
        output=attempt.get('output',{});syntax=False;shape=False;good={};bad={}
        try:
            value=strict_json(output.get('text',''));syntax=True;shape=schema_ok(value,attempt['schema'])
            if type(value) is dict and value.get('agent_role')==key[2] and type(value.get('features')) is list:
                for item in value['features']:
                    if type(item) is dict and item.get('feature_id') in attempt['features']:emitted[key].add(item['feature_id'])
        except (ValueError,TypeError):pass
        if output.get('finish_reason')=='stop':
            good,bad=validate(output.get('text',''),key[2],attempt['features'],source_fields(mail,chunk))
        attempt_metrics.append(dict(source=key[0],id=key[1],role=key[2],initial=first,syntax_valid=syntax,schema_valid=shape,
            assigned=len(attempt['features']),valid_features=len(good),full_contract_valid=not bad and len(good)==len(attempt['features']) and output.get('finish_reason')=='stop',
            finish_reason=output.get('finish_reason','error')))
    def summarize(roles,tries):
        slots=sum(len(feature_ids(k[2])) for k,_ in roles)
        fields=[f for _,r in roles for f in r['features']]
        valid=sum(f['status']=='valid' for f in fields)
        grounded=sum(f['status']=='valid' and f['result']['found'] for f in fields)
        emitted_n=sum(len(emitted[k]) for k,_ in roles)
        initial=[x for x in tries if x['initial']]
        return dict(role_reports=len(roles),assigned_feature_slots=slots,generation_attempts=len(tries),initial_attempts=len(initial),retries=len(tries)-len(initial),
            json_syntax_valid=sum(x['syntax_valid'] for x in tries),json_validity=ratio(sum(x['syntax_valid'] for x in tries),len(tries)),
            json_validity_initial=ratio(sum(x['syntax_valid'] for x in initial),len(initial)),
            schema_valid=sum(x['schema_valid'] for x in tries),schema_validity=ratio(sum(x['schema_valid'] for x in tries),len(tries)),
            initial_full_contract_valid=sum(x['full_contract_valid'] for x in initial),
            final_full_contract_valid=sum(r['status']=='valid' for _,r in roles),final_full_contract_rate=ratio(sum(r['status']=='valid' for _,r in roles),len(roles)),
            emitted_feature_slots=emitted_n,emitted_feature_coverage=ratio(emitted_n,slots),valid_feature_slots=valid,valid_feature_coverage=ratio(valid,slots),
            found_grounded_feature_slots=grounded,found_grounded_feature_fraction=ratio(grounded,slots),
            reports_without_valid_nonempty_quote=sum(not any(f['status']=='valid' and f['result']['found'] for f in r['features']) for _,r in roles),
            role_status_counts=dict(collections.Counter(r['status'] for _,r in roles)),
            finish_reason_counts=dict(collections.Counter(x['finish_reason'] for x in tries)))
    roles=list(final_records.items())
    return dict(overall=summarize(roles,attempt_metrics),
        by_source={s:summarize([(k,r) for k,r in roles if k[0]==s],[a for a in attempt_metrics if a['source']==s]) for s in SOURCES},
        by_role={role:summarize([(k,r) for k,r in roles if k[2]==role],[a for a in attempt_metrics if a['role']==role]) for role in ROLES},
        mails=len(final),all_four_contract_valid_mails=sum(all(r['status']=='valid' for r in e['evidence']['records']) for e in final),
        evidence_accuracy=None,role_consistency=None,semantic_status='not_estimable_without_complete_semantic_annotations',attempt_audit=attempt_metrics)

def calculate(archive):
    with tarfile.open(archive) as tar:
        read=lambda name:[json.loads(s) for s in tar.extractfile(PREFIX+'/'+name) if s.strip()]
        truth={(source,r['id']):r['label'] for source in SOURCES for r in read('data/'+source+'/labels.jsonl')}
        compact=[];signatures=set();failures=[]
        for row in read('scores.jsonl'):
            assert row['record_sha256']==digest({k:v for k,v in row.items() if k!='record_sha256'})
            signatures.add(row['run_signature']);key=(row['source'],row['id']);y=truth[key]
            p=row['raw_equal_mean'];roles={r['role']:r['normalized_label_score'] for r in row['roles']}
            if p is not None:assert len(roles)==4 and math.isclose(p,sum(roles.values())/4,abs_tol=1e-12)
            else:failures.append(dict(source=key[0],id=key[1],label=y,reason=row['reason']))
            compact.append(dict(source=key[0],id=key[1],y=y,p=p,roles=roles))
        assert len(compact)==len(truth)==2000 and len({(r['source'],r['id']) for r in compact})==2000 and len(signatures)==1
        cohorts={s:[r for r in compact if r['source']==s] for s in SOURCES}|{'pooled':compact}
        classification={s:metrics(rows) for s,rows in cohorts.items()}
        agents={s:{role:metrics([dict(r,p=r['roles'].get(role)) for r in rows]) for role in ROLES} for s,rows in cohorts.items()}
        mutual={s:complementary(rows) for s,rows in cohorts.items()}
        evidence=evidence_metrics(tar)
    return dict(version='v5-basic-metrics-20261030',provenance=dict(archive=str(archive.resolve()),archive_sha256=digest(archive.read_bytes()),
        analysis_script_sha256=digest(Path(__file__).read_bytes()),run_signature=next(iter(signatures))),
        settings=dict(threshold=.5,calibration=False,ppo=False,score='uncalibrated mean of four sigmoid margins',
            accuracy='accuracy_all uses all samples; an abstention is not correct',
            classification_rates='precision/recall/F1/FNR/HAM recall/FPR use scored cases; abstentions listed separately',
            probability_metrics='Brier and positive-probability ECE use scored cases only',
            ece='10 equal-width bins on p; sum(n_bin/n * abs(mean(p)-mean(y)))',
            correlation='Pearson correlation of binary prediction-error indicators on common scored cases; undefined for zero variance',
            joint_error_rate='both agents wrong / common scored cases',
            feature_coverage='emitted unique assigned IDs across parseable attempts / assigned slots; valid coverage separately applies schema and quote checks',
            json_validity='strict JSON syntax per raw generation attempt, including retries; role schema and final merged contract separately reported',
            pooled='descriptive pooling of different native spam/phishing tasks; interpret source-level metrics first',
            roc_auc='additional rank diagnostic; no threshold tuning'),
        classification=classification,agents=agents,complementarity=mutual,evidence=evidence,failures=failures,
        ppo_gain=dict(value=None,status='not_run'),
        formulas=dict(accuracy='(TP+TN)/N_all',precision='TP/(TP+FP)',recall='TP/(TP+FN)',f1='2TP/(2TP+FP+FN)',
            fnr='FN/(TP+FN)',ham_recall='TN/(TN+FP)',fpr='FP/(TN+FP)',brier='mean((p-y)^2)'),
        undefined='null means undefined or unavailable; no positive predictions makes precision undefined, not zero')

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=calculate(args.archive);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(classification={s:{k:v for k,v in m.items() if k!='reliability'} for s,m in result['classification'].items()},evidence=result['evidence']['overall']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
