import math
from dataclasses import dataclass
from ..contracts import ROLES, RawRoleScore, CalibratedRoleScore
from ..util import digest,sigmoid
from ..contracts.domain import DomainPolicy

@dataclass(frozen=True)
class AffineCalibrator:
    role: str
    source: str
    scoring_signature: str
    center: float
    scale: float
    slope: float
    intercept: float

    def __post_init__(self):
        if self.role not in ROLES or not all(math.isfinite(v) for v in (self.center,self.scale,self.slope,self.intercept)) or self.scale<=0 or self.slope<=0:
            raise ValueError('invalid monotone calibrator')

    def probability(self,margin):return sigmoid(self.slope*(margin-self.center)/self.scale+self.intercept)

def fit_affine(margins,labels,role,source,signature,regularization=.05):
    import torch
    if len(margins)!=len(labels) or len(margins)<4 or set(labels)!={0,1}:
        raise ValueError('calibration requires both classes and at least four rows')
    if regularization<=0:raise ValueError('positive regularization required')
    x=torch.tensor(margins,dtype=torch.float64); y=torch.tensor(labels,dtype=torch.float64)
    if not torch.isfinite(x).all():raise ValueError('nonfinite margins')
    center=x.mean().item();scale=max(x.std(unbiased=False).item(),1e-6);z=(x-center)/scale
    raw_slope=torch.tensor(.5413248546,dtype=torch.float64,requires_grad=True)
    bias=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([raw_slope,bias],lr=.5,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();a=torch.nn.functional.softplus(raw_slope)+1e-6
        loss=torch.nn.functional.binary_cross_entropy_with_logits(a*z+bias,y)+regularization*((a-1).square()+bias.square())
        loss.backward();return loss
    optimizer.step(closure)
    return AffineCalibrator(role,source,signature,center,scale,
        (torch.nn.functional.softplus(raw_slope)+1e-6).item(),bias.item())

@dataclass(frozen=True)
class CalibratorSet:
    source: str
    scoring_signature: str
    models: tuple[AffineCalibrator,...]
    fit_members: tuple[dict,...]
    domain_policy: DomainPolicy
    protocol: str = 'domain-bound-monotone-affine-v3.0'

    def __post_init__(self):
        if tuple(m.role for m in self.models)!=ROLES:raise ValueError('canonical four-role calibration required')
        if any(m.source!=self.source or m.scoring_signature!=self.scoring_signature for m in self.models):raise ValueError('calibrator binding mismatch')
        if self.domain_policy.label_schema!=self.source:raise ValueError('domain label binding mismatch')

    @property
    def signature(self):return digest(self)

    def apply(self,raw):
        if type(raw) is not RawRoleScore:raise TypeError('RawRoleScore required')
        if raw.source!=self.source or raw.scoring_signature!=self.scoring_signature:raise ValueError('calibrator source/scoring signature mismatch')
        if not self.domain_policy.accepts(raw):raise ValueError('calibrator is not applicable to score domain')
        if not math.isclose(raw.coverage,1.,abs_tol=1e-8):raise ValueError('incomplete input coverage')
        model=self.models[ROLES.index(raw.role)]
        return CalibratedRoleScore(raw,model.probability(raw.margin),self.signature)

    @classmethod
    def from_dict(cls,value):
        return cls(value['source'],value['scoring_signature'],tuple(AffineCalibrator(**m) for m in value['models']),tuple(value['fit_members']),DomainPolicy.from_dict(value['domain_policy']),value['protocol'])
