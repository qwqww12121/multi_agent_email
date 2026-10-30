import math
from ..contracts import ROLES,CalibratedRoleScore

def probabilities(scores,source=None,calibrator_signature=None):
    if len(scores)!=4 or any(type(s) is not CalibratedRoleScore for s in scores):
        raise TypeError('fusion accepts four calibrated role scores; raw scores forbidden')
    if tuple(s.raw.role for s in scores)!=ROLES:raise ValueError('canonical role order required')
    if any(len({getattr(s.raw,k) for s in scores})!=1 for k in ('input_signature','source_dataset','task_type','calibration_domain','input_protocol','scoring_signature')):raise ValueError('roles belong to different inputs or scoring domains')
    if len({s.raw.source for s in scores})!=1 or len({s.calibrator_signature for s in scores})!=1:
        raise ValueError('incompatible calibrated scores')
    if source is not None and any(s.raw.source!=source for s in scores):raise ValueError('fusion source mismatch')
    if calibrator_signature is not None and any(s.calibrator_signature!=calibrator_signature for s in scores):
        raise ValueError('fusion calibrator signature mismatch')
    return [s.probability for s in scores]

def state(p,quality):
    from ..experts.quality import validate_quality, QUALITY_SIZE
    validate_quality(quality,4*QUALITY_SIZE)
    m=sum(p)/4
    return [*p,m,math.sqrt(sum((x-m)**2 for x in p)/4),max(p)-min(p),*quality]

def convex(p,w):
    if len(w)!=4 or any(not math.isfinite(x) or x<0 for x in w) or not math.isclose(sum(w),1.,abs_tol=1e-6):raise ValueError('invalid simplex weights')
    return sum(a*b for a,b in zip(p,w))

def validate_matrix(values,labels):
    if len(values)!=len(labels) or len(values)<4 or set(labels)!={0,1}:raise ValueError('both classes and four rows required')
    if any(len(r)!=4 or any(not math.isfinite(v) or not 0<=v<=1 for v in r) for r in values):raise ValueError('four bounded probabilities per row required')
