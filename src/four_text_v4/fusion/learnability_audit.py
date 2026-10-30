from .common import validate_matrix

def audit(values,labels,threshold=.5):
    validate_matrix(values,labels)
    details=[]
    for p,y in zip(values,labels):
        lo,hi=min(p),max(p)
        details.append({'min':lo,'max':hi,'classification_can_change':lo<threshold<=hi,
            'brier_reward_range':abs((lo-y)**2-(hi-y)**2)})
    return {'n':len(values),'classification_changeable':sum(r['classification_can_change'] for r in details),
            'continuous_reward_changeable':sum(r['brier_reward_range']>1e-12 for r in details),
            'threshold':threshold,'details':details}
