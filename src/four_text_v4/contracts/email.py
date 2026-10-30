from dataclasses import dataclass
from .task import TASKS
from .domain import task_for,INPUT_PROTOCOL

@dataclass(frozen=True)
class RawEmail:
    id: str
    label_schema: str
    source_dataset: str
    task_type: str
    calibration_domain: str
    subject: str = ''
    sender: str = ''
    body: str = ''
    html: str = ''
    mime: str = ''
    family_id: str = ''
    input_protocol: str = INPUT_PROTOCOL

    @property
    def source(self):return self.label_schema

    def __post_init__(self):
        if not self.id or not self.source_dataset or not self.calibration_domain:raise ValueError('explicit identity, actual source and calibration domain required')
        if self.task_type!=task_for(self.label_schema) or self.input_protocol!=INPUT_PROTOCOL:raise ValueError('task/input protocol mismatch')
        if any(not isinstance(v,str) for v in vars(self).values()): raise TypeError('raw email fields must be strings')
        if self.mime and any((self.subject,self.sender,self.body,self.html)):raise ValueError('MIME must be the sole message representation; explicit fields would be ambiguous')

    @classmethod
    def from_dict(cls, row):
        # Labels and prepared records must be explicitly adapted outside inference.
        if set(row)-set(cls.__dataclass_fields__): raise ValueError('unknown raw-email fields; labels/prepared input forbidden')
        return cls(**row)

@dataclass(frozen=True)
class Block:
    id: str
    kind: str
    start: int
    end: int
    text: str

@dataclass(frozen=True)
class CanonicalEmail:
    raw: RawEmail
    subject: str
    sender: str
    body: str
    blocks: tuple[Block,...]
    links: tuple[tuple[str,str],...]
    attachments: tuple[str,...]
    missing: tuple[str,...]
    fatal: str | None
    body_hash: str
    parser_version: str

@dataclass(frozen=True)
class ModelInput:
    mail_id: str
    index: int
    total: int
    body_start: int
    body_end: int
    text: str
    blocks: tuple[Block,...]
    coverage: float
    protocol: str
