"""Optional independent evidence branch; invalid features stay unavailable."""
from dataclasses import asdict, dataclass
from pathlib import Path
from ..contracts import ROLES
from ..contracts.evidence import schema,validate,VERSION
from ..experts.catalog import feature_ids,load_role,resource
from ..ingestion import InputBudget,plan_inputs
from ..util import digest
from .render import render_evidence,source_fields

@dataclass(frozen=True)
class EvidenceConfig:
    max_new_tokens:int=1024
    max_retries:int=1
    quote_mode:str='exact'

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens<64:raise ValueError('positive evidence budget required')
        if type(self.max_retries) is not int or not 0<=self.max_retries<=2:raise ValueError('retry budget must be 0..2')
        if self.quote_mode not in ('exact','whitespace'):raise ValueError('invalid quote mode')

class EvidenceAnalyzer:
    def __init__(self,backend,budget=InputBudget(),config=EvidenceConfig()):
        if not callable(getattr(backend,'generate_evidence',None)):raise TypeError('evidence generation backend required')
        self.backend,self.budget,self.config=backend,budget,config
        root=Path(__file__).resolve().parents[1]
        self.signature=digest(dict(version=VERSION,backend=backend.identity,budget=asdict(budget),config=asdict(config),
            decoder='lm-format-enforcer-0.11.3',template=resource('common/evidence_base.txt'),roles={r:load_role(r) for r in ROLES},
            code={n:digest((root/n).read_bytes()) for n in ('scoring/evidence.py','scoring/render.py','scoring/json_transport.py','contracts/evidence.py','experts/catalog.py')}))

    def analyze(self,mail):
        if mail.fatal:raise ValueError('unusable evidence input: '+mail.fatal)
        limit=min(self.budget.max_tokens,getattr(self.backend,'max_tokens',self.budget.max_tokens))
        def fits(chunk):
            return all(len(self.backend.tokenizer.encode(render_evidence(mail,chunk,r,retry=True)[0],add_special_tokens=False))+self.config.max_new_tokens<=limit for r in ROLES)
        chunks=plan_inputs(mail,self.budget,fits)
        records=[]
        for chunk in chunks:
            for role in ROLES:
                assigned=feature_ids(role);pending=list(assigned);accepted={};attempts=[];errors={}
                for retry in range(self.config.max_retries+1):
                    prompt,prefix=render_evidence(mail,chunk,role,pending,retry>0)
                    answer={};good={}
                    try:
                        answer=self.backend.generate_evidence(prompt,self.config.max_new_tokens,schema(role,pending),prefix)
                        if answer['finish_reason']!='stop':raise ValueError('generation length limit')
                        if answer.get('trailing_text','').strip():raise ValueError('unexpected text after JSON object')
                        good,errors=validate(answer['text'],role,pending,source_fields(mail,chunk),self.config.quote_mode)
                    except (ValueError,RuntimeError,TypeError,KeyError) as exc:errors={fid:str(exc) for fid in pending}
                    attempts.append(dict(retry=retry,feature_ids=list(pending),prompt=prompt,prompt_hash=digest(prompt),
                        output=answer,errors=dict(errors)))
                    accepted.update(good)
                    pending=[fid for fid in pending if fid not in good]
                    if not pending:break
                records.append(dict(role=role,chunk=chunk.index,status='valid' if not pending else 'partial' if accepted else 'unreliable',
                    features=[dict(feature_id=fid,status='valid' if fid in accepted else 'unreliable',
                        **accepted.get(fid,dict(result=None,error=errors.get(fid,'unavailable')))) for fid in assigned],attempts=attempts))
        return dict(signature=self.signature,config=asdict(self.config),chunks=[asdict(c) for c in chunks],records=records,
            limitation='Validation establishes schema and quote membership only; found/suspicious and role relevance require semantic assessment.')
