from ..util import digest

def evaluate(predictions,labels):
    truth={(r['source_dataset'],r['id']):r['label'] for r in labels}
    if len(truth)!=len(labels) or any(type(y) is not int or y not in (0,1) for y in truth.values()):raise ValueError('unique binary labels required')
    found={(r['source_dataset'],r['id']):r for r in predictions}
    if len(found)!=len(predictions) or set(found)-set(truth):raise ValueError('duplicate/unexpected predictions')
    for r in predictions:
        if r.get('record_sha256')!=digest({k:v for k,v in r.items() if k!='record_sha256'}):raise ValueError('prediction record hash mismatch')
        if r.get('method')!='ppo':raise ValueError('only PPO predictions supported')
    if len({r['run_signature'] for r in predictions})>1:raise ValueError('mixed runs must be evaluated separately')
    output={}
    for source in sorted({k[0] for k in truth}):
        cells={'n':0,'tp':0,'tn':0,'fp':0,'fn':0,'unknown':0,'missing':0};errors=[];probability_pairs=[]
        for key,y in truth.items():
            if key[0]!=source:continue
            cells['n']+=1;r=found.get(key)
            if r is None:cells['missing']+=1;continue
            if r['status']!='ok':cells['unknown']+=1;continue
            p=r['prediction'];prob=r['probability']
            if p not in (0,1) or not 0<=prob<=1:raise ValueError('invalid prediction')
            cells['tp' if p and y else 'fp' if p else 'fn' if y else 'tn']+=1;errors.append((prob-y)**2)
            probability_pairs.append((prob,y))
        cells['accuracy_all']=(cells['tp']+cells['tn'])/cells['n']
        cells['coverage']=len(errors)/cells['n'];cells['brier_successful_only']=sum(errors)/len(errors) if errors else None
        cells['positive_support']=sum(y==1 for key,y in truth.items() if key[0]==source)
        cells['negative_support']=cells['n']-cells['positive_support']
        cells['recall_all_positive']=cells['tp']/cells['positive_support'] if cells['positive_support'] else None
        cells['specificity_all_negative']=cells['tn']/cells['negative_support'] if cells['negative_support'] else None
        cells['precision_successful_only']=cells['tp']/(cells['tp']+cells['fp']) if cells['tp']+cells['fp'] else None
        denominator=2*cells['tp']+cells['fp']+cells['fn']
        cells['f1_successful_only']=2*cells['tp']/denominator if denominator else None
        cells['fnr_all_positive']=1-cells['recall_all_positive'] if cells['positive_support'] else None
        cells['fpr_all_negative']=cells['fp']/cells['negative_support'] if cells['negative_support'] else None
        bins=[[] for _ in range(10)]
        for p,y in probability_pairs:bins[min(9,int(p*10))].append((p,y))
        cells['ece10_successful_only']=sum(abs(sum(p-y for p,y in b)) for b in bins)/len(errors) if errors else None
        cells['ece_definition']='10 equal-width positive-probability bins; successful predictions only'
        output[source]=cells
    return output
