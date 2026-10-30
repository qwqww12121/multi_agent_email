from dataclasses import dataclass
import math
from .task import TASKS

INPUT_PROTOCOL='canonical-visible-email-v5.0'
def task_for(schema):
    if schema not in TASKS:raise ValueError('unknown native label schema')
    return 'spam' if schema in ('ceas_2008_curated','enron_spam') else 'phishing'

@dataclass(frozen=True)
class DomainPolicy:
    name: str
    task_type: str
    label_schema: str
    source_datasets: tuple[str,...]
    validation_scope: str = 'within_source'
    input_protocol: str = INPUT_PROTOCOL
    threshold: float = .5
    min_auc: float = .5
    min_slope: float = .05
    min_probability_spread: float = .01
    min_class_count: int = 4

    def __post_init__(self):
        if any(not math.isfinite(v) for v in (self.threshold,self.min_auc,self.min_slope,self.min_probability_spread)):raise ValueError('finite domain settings required')
        if not self.name or self.task_type!=task_for(self.label_schema):raise ValueError('task/label schema mismatch')
        if isinstance(self.source_datasets,str) or not self.source_datasets or len(set(self.source_datasets))!=len(self.source_datasets) or any(not isinstance(s,str) or not s.strip() for s in self.source_datasets):raise ValueError('explicit unique real source datasets required')
        if self.validation_scope not in ('within_source','declared_transfer'):raise ValueError('explicit validation scope required')
        if self.input_protocol!=INPUT_PROTOCOL:raise ValueError('unsupported input protocol')
        if not 0<self.threshold<1 or not .5<=self.min_auc<=1 or self.min_slope<=0 or not 0<self.min_probability_spread<1 or self.min_class_count<2:raise ValueError('invalid fixed acceptance policy')

    def accepts(self,row):
        return (row.source_dataset in self.source_datasets and row.task_type==self.task_type
                and row.source==self.label_schema and row.calibration_domain==self.name
                and row.input_protocol==self.input_protocol)

    @classmethod
    def from_dict(cls,value):
        value=dict(value);value['source_datasets']=tuple(value['source_datasets']);return cls(**value)
