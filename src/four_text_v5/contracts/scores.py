import math
from dataclasses import dataclass
from .task import ROLES, TASKS

@dataclass(frozen=True)
class CandidateObservation:
    chunk: int
    definition_order: str
    prompt: str
    prompt_hash: str
    negative_ll: float
    positive_ll: float
    prompt_tokens: int
    candidate_tokens: tuple[int,int]
    summed_logprobs: tuple[float,float]
    normalization: str

    @property
    def margin(self): return self.positive_ll-self.negative_ll

@dataclass(frozen=True)
class RawRoleScore:
    role: str
    source: str
    scoring_signature: str
    candidates: tuple[str,str]
    observations: tuple[CandidateObservation,...]
    margin: float
    normalized_label_score: float
    order_sensitivity: float
    coverage: float
    source_dataset: str
    task_type: str
    calibration_domain: str
    input_protocol: str
    input_signature: str

    def __post_init__(self):
        if self.role not in ROLES or self.source not in TASKS: raise ValueError('invalid role/source')
        if self.candidates!=TASKS[self.source].candidates: raise ValueError('native candidates mismatch')
        from .domain import task_for,INPUT_PROTOCOL
        if not self.source_dataset or not self.calibration_domain or self.task_type!=task_for(self.source) or self.input_protocol!=INPUT_PROTOCOL:raise ValueError('score domain binding required')
        if not all(math.isfinite(x) for x in (self.margin,self.normalized_label_score,self.order_sensitivity,self.coverage)):
            raise ValueError('nonfinite raw score')
        if not 0<=self.normalized_label_score<=1 or not 0<=self.coverage<=1: raise ValueError('invalid score range')
        from ..util import sigmoid
        if not self.scoring_signature or not self.input_signature:raise ValueError('score/input signature required')
        if not math.isclose(self.normalized_label_score,sigmoid(self.margin),abs_tol=1e-12):raise ValueError('normalized score does not match two-label likelihood margin')

@dataclass(frozen=True)
class CalibratedRoleScore:
    raw: RawRoleScore
    probability: float
    calibrator_signature: str

    def __post_init__(self):
        if not isinstance(self.raw,RawRoleScore) or not self.calibrator_signature: raise TypeError('bound raw score/calibrator required')
        if not math.isfinite(self.probability) or not 0<=self.probability<=1: raise ValueError('invalid calibrated probability')
