"""Classification never depends on generated evidence or an output JSON parser."""
import math
import importlib.metadata
from dataclasses import asdict, dataclass
from pathlib import Path
from ..contracts import ROLES, TASKS, RawRoleScore, CandidateObservation
from ..experts.catalog import load_role, resource
from ..ingestion import InputBudget, plan_inputs
from ..util import digest, sigmoid
from .render import render_classification

ORDERS = ('negative_first', 'positive_first')

@dataclass(frozen=True)
class ScoringConfig:
    normalization: str = 'mean_token'

    def __post_init__(self):
        if self.normalization not in ('mean_token', 'sum'):
            raise ValueError('unknown candidate normalization')

class RoleScoringFailure(RuntimeError):
    def __init__(self, role, chunk, error, completed, partial):
        super().__init__(f'{role}/chunk={chunk}: {type(error).__name__}: {error}')
        self.completed, self.partial = tuple(completed), tuple(partial)

class RoleScorer:
    def __init__(self, backend, budget=InputBudget(), config=ScoringConfig()):
        if not callable(getattr(backend,'score',None)):raise TypeError('candidate scoring backend required')
        self.backend, self.budget, self.config = backend, budget, config
        root=Path(__file__).resolve().parents[1]
        names=['scoring/role_scorer.py','scoring/render.py','scoring/candidate_backend.py',
               'experts/catalog.py','contracts/scores.py','contracts/email.py','contracts/task.py','contracts/domain.py','util.py']
        names += [p.relative_to(root).as_posix() for p in (root/'ingestion').glob('*.py')]
        versions={}
        for name in ('torch','transformers','tokenizers'):
            try:versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:versions[name]=None
        self.signature=digest(dict(version='classification-v5',backend=backend.identity,
            budget=asdict(budget),config=asdict(config),orders=ORDERS,versions=versions,
            aggregation='two-order mean margins; complete chunks weighted by body characters',
            template=resource('common/classification_base.txt'),roles={r:load_role(r) for r in ROLES},
            code={n:digest((root/n).read_bytes()) for n in sorted(names)}))

    def _tokens(self,text):return len(self.backend.tokenizer.encode(text,add_special_tokens=False))

    def score(self,mail):
        if mail.fatal:raise ValueError('unusable input: '+mail.fatal)
        candidates=TASKS[mail.raw.source].candidates
        limit=min(self.budget.max_tokens,getattr(self.backend,'max_tokens',self.budget.max_tokens))
        tail=max(self._tokens(' '+c) for c in candidates)
        chunks=plan_inputs(mail,self.budget,lambda c:all(
            self._tokens(render_classification(mail,c,r,o))+tail<=limit for r in ROLES for o in ORDERS))
        scores=[]
        for role in ROLES:
            observations=[];margin=0.;sensitivity=0.
            for chunk in chunks:
                pair=[]
                for order in ORDERS:
                    prompt=render_classification(mail,chunk,role,order)
                    try:
                        answer=self.backend.score(prompt,candidates)
                        sums=tuple(answer['logprobs']);counts=tuple(answer['candidate_token_counts'])
                        if len(sums)!=2 or any(not math.isfinite(x) or x>0 for x in sums):raise ValueError('invalid conditional log likelihood')
                        expected=tuple(self._tokens(' '+c) for c in candidates)
                        if counts!=expected or any(type(n) is not int or n<1 for n in counts):raise ValueError('candidate token counts mismatch')
                        if answer['prompt_tokens']!=self._tokens(prompt):raise ValueError('prompt token count mismatch')
                        ll=tuple(v/n if self.config.normalization=='mean_token' else v for v,n in zip(sums,counts))
                        obs=CandidateObservation(chunk.index,order,prompt,digest(prompt),*ll,answer['prompt_tokens'],counts,sums,self.config.normalization)
                        observations.append(obs);pair.append(obs.margin)
                    except (ValueError,RuntimeError,TypeError,KeyError) as exc:
                        raise RoleScoringFailure(role,chunk.index,exc,scores,observations) from exc
                margin+=sum(pair)/2*chunk.coverage
                sensitivity+=abs(pair[0]-pair[1])*chunk.coverage
            scores.append(RawRoleScore(role,mail.raw.source,self.signature,candidates,tuple(observations),
                margin,sigmoid(margin),sensitivity,min(1.,math.fsum(c.coverage for c in chunks)),
                mail.raw.source_dataset,mail.raw.task_type,mail.raw.calibration_domain,mail.raw.input_protocol,
                digest(asdict(mail))))
        return tuple(scores),chunks
