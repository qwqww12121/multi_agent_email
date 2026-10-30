import difflib,math,unicodedata
from dataclasses import dataclass
from ..util import digest
from ..contracts.domain import task_for,INPUT_PROTOCOL

def normalized_hash(body):return digest(' '.join(unicodedata.normalize('NFKC',body).casefold().split()))

@dataclass(frozen=True)
class ExclusionRow:
    id: str
    label_schema: str
    source_dataset: str
    task_type: str
    calibration_domain: str
    body: str
    body_hash: str
    family_id: str
    content_hashes: tuple[str,...]
    input_protocol: str = INPUT_PROTOCOL

    @property
    def source(self):return self.label_schema

    def __post_init__(self):
        if not self.id or not self.source_dataset or not self.calibration_domain:raise ValueError('explicit source identity required')
        if self.task_type!=task_for(self.label_schema) or self.input_protocol!=INPUT_PROTOCOL:raise ValueError('task/input protocol mismatch')
        if self.body_hash!=normalized_hash(self.body):raise ValueError('canonical body/hash mismatch')
        if any(not isinstance(h,str) or len(h)!=64 for h in self.content_hashes):raise ValueError('invalid content view hash')

    def member(self):
        return dict(id=self.id,source=self.source_dataset,body_hash=self.body_hash,family_id=self.family_id,content_hashes=list(self.content_hashes))

@dataclass(frozen=True)
class TrainingRow(ExclusionRow):
    label: int = -1
    margins: tuple[float,...] = ()
    scoring_signature: str = ''
    evidence_quality: tuple[float,...] = ()

    def __post_init__(self):
        super().__post_init__()
        from ..experts.quality import validate_quality, QUALITY_SIZE
        validate_quality(self.evidence_quality,4*QUALITY_SIZE)
        if type(self.label) is not int or self.label not in (0,1):raise ValueError('binary label required')
        if not self.scoring_signature or not self.body.strip() or len(self.margins)!=4 or not all(math.isfinite(v) for v in self.margins):raise ValueError('four finite margins and scoring signature required')

def overlap(a,b):
    return ((a['source'],a['id'])==(b['source'],b['id']) or a['body_hash']==b['body_hash']
        or bool(a.get('family_id') and a.get('family_id')==b.get('family_id'))
        or bool(set(a.get('content_hashes',()))&set(b.get('content_hashes',()))))

def assert_unseen(member,trained):
    if any(overlap(member,m) for m in trained):raise ValueError('input overlaps fitting or validation data by identity/content/family')

def audit_splits(splits,near_threshold=.95):
    if not 0<near_threshold<=1 or any(not r for r in splits.values()):raise ValueError('nonempty splits and valid threshold required')
    for rows in splits.values():
        keys=[(r.source_dataset,r.id) for r in rows]
        if len(set(keys))!=len(keys):raise ValueError('duplicate source identity within split')
    checks=0;names=list(splits)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for x in splits[a]:
                for y in splits[b]:
                    checks+=1
                    if overlap(x.member(),y.member()):raise ValueError(f'split overlap: {a}/{b}')
                    m=difflib.SequenceMatcher(None,' '.join(x.body.casefold().split()),' '.join(y.body.casefold().split()),autojunk=False)
                    if m.real_quick_ratio()>=near_threshold and m.quick_ratio()>=near_threshold and m.ratio()>=near_threshold:raise ValueError(f'near-duplicate split overlap: {a}/{b}')
    return dict(counts={k:len(v) for k,v in splits.items()},pairs_checked=checks,near_threshold=near_threshold,
        scope='all cross-split canonical bodies plus source identities, known families and nonempty text views; no semantic deduplication',
        manifest_signature=digest({k:[r.member() for r in v] for k,v in splits.items()}))

def validate_training_rows(rows,policy):
    if not rows or len({r.scoring_signature for r in rows})!=1 or any(not policy.accepts(r) for r in rows):raise ValueError('mixed scoring contract or undeclared domain/source/task')
    if set(r.label for r in rows)!={0,1}:raise ValueError('both classes required')
