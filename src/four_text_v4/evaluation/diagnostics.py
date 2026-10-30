import math
from ..contracts import ROLES

def auc(values,labels):
    positive=[v for v,y in zip(values,labels) if y==1];negative=[v for v,y in zip(values,labels) if y==0]
    if not positive or not negative:return None
    return sum(1. if p>n else .5 if p==n else 0. for p in positive for n in negative)/(len(positive)*len(negative))

def role_diagnostics(margins,probabilities,labels):
    """Descriptive only; no accuracy-based experiment rejection or threshold search."""
    report={}
    for i,role in enumerate(ROLES):
        m=[r[i] for r in margins];p=[r[i] for r in probabilities]
        report[role]={'raw_margin_auc':auc(m,labels),'calibrated_auc':auc(p,labels),
            'brier':sum((x-y)**2 for x,y in zip(p,labels))/len(labels),
            'positive_predictions':sum(x>=.5 for x in p),'margin_min':min(m),'margin_max':max(m)}
    agreement={}
    for i,r in enumerate(ROLES):
        for j in range(i+1,len(ROLES)):
            agreement[f'{r}/{ROLES[j]}']=sum((p[i]>=.5)==(p[j]>=.5) for p in probabilities)/len(labels)
    return {'roles':report,'pairwise_class_agreement':agreement,'n':len(labels),
        'interpretation':'Training diagnostic, not heldout efficacy. Calibration cannot create missing ranking information.'}
