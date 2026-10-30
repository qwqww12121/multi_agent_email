from dataclasses import dataclass
from ..contracts import CanonicalEmail, ModelInput, Block

class BudgetExceeded(ValueError): pass

@dataclass(frozen=True)
class InputBudget:
    max_tokens: int = 18432
    max_chunks: int = 16
    mode: str = 'reject'

    def __post_init__(self):
        if self.max_tokens<64 or self.max_chunks<1 or self.mode not in ('reject','chunk_all'):
            raise ValueError('invalid budget')

def slice_input(mail,start,end,index=0,total=1):
    refs=tuple(Block(p.id,p.kind,max(start,p.start),min(end,p.end),mail.body[max(start,p.start):min(end,p.end)])
               for p in mail.blocks if p.start<end and p.end>start)
    return ModelInput(mail.raw.id,index,total,start,end,mail.body[start:end],refs,(end-start)/max(1,len(mail.body)),'complete-chunks-v4.0')

def plan_inputs(mail: CanonicalEmail, budget: InputBudget, fits):
    """fits renders ALL roles/orders including complete candidates; no approximate budget."""
    whole=slice_input(mail,0,len(mail.body))
    if fits(whole):return (whole,)
    if budget.mode=='reject':raise BudgetExceeded('complete input exceeds budget; no truncation')
    # Deterministic halving; all characters, including block-crossing context, retained.
    pending=[(0,len(mail.body))]; accepted=[]
    while pending:
        start,end=pending.pop(0);chunk=slice_input(mail,start,end)
        if fits(chunk):accepted.append((start,end));continue
        if end-start<=1:raise BudgetExceeded('metadata/evidence overhead exceeds budget')
        middle=(start+end)//2
        pending[0:0]=[(start,middle),(middle,end)]
        if len(accepted)+len(pending)>budget.max_chunks:raise BudgetExceeded('chunk limit exceeded; entire mail is Unknown')
    accepted.sort()
    result=tuple(slice_input(mail,s,e,i,len(accepted)) for i,(s,e) in enumerate(accepted))
    if ''.join(c.text for c in result)!=mail.body:raise AssertionError('coverage gap')
    if not all(fits(c) for c in result):raise BudgetExceeded('final chunk metadata exceeds budget')
    return result
