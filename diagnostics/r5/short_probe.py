"""Adaptive follow-up: no model-authored JSON; two short completion styles."""
import os,json,time,hashlib
from pathlib import Path
from probe import HFScorer,torch,StoppingCriteria,StoppingCriteriaList,sha
ROOT = Path(os.environ.get('MAIL_AGENT_RUN_DIR', str(Path(__file__).resolve().parent))).resolve()
OUT=ROOT/'short_followup'
QUESTIONS={
 'action':('What does the sender ask the recipient to do?','The sender asks the recipient to'),
 'purpose':('What is the main topic of this email?','The main topic of this email is'),
 'persuasion':('How does the sender try to persuade the recipient?','The sender tries to persuade the recipient by'),
 'context':('Who wrote this email? Use the signature or self-introduction, not the recipient name.','The sender of this email is')}
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 OUT.mkdir(exist_ok=False)
 config={'version':'r5b-short-qa-cloze','max_new_tokens':80,'do_sample':False,'questions':QUESTIONS,'script_sha256':sha(__file__),'inputs_sha256':sha(ROOT/'inputs.jsonl'),'scope':'adaptive diagnostic on same 4 mails; program wraps text; no claim of full-report success','eos_ids':[128001,128008,128009],'stop':'first newline after non-whitespace answer or eos or budget'}
 signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();save('config.json',config)
 save('status.json',{'phase':'loading'})
 b=HFScorer(os.environ['LLAMA_MODEL_PATH']);tok,model=b.tokenizer,b.model
 save('runtime.json',{'model':b.identity,'signature':signature})
 mails=[json.loads(l) for l in (ROOT/'inputs.jsonl').read_text(encoding='utf-8').splitlines()]
 count=0;start=time.time()
 with (OUT/'records.jsonl').open('x',encoding='utf-8',buffering=1) as out:
  for style in ['qa','cloze']:
   for role,(question,stem) in QUESTIONS.items():
    prompts=[]
    for m in mails:
     p=tok.bos_token+'Email:\n'+m['body']+'\n\n'
     if style=='qa':p+='Question: '+question+' Answer briefly using only the email.\nAnswer:'
     else:p+=stem
     prompts.append(p)
    rows=[tok.encode(p,add_special_tokens=False) for p in prompts];w=max(map(len,rows));pad=128001
    ids=torch.tensor([[pad]*(w-len(r))+r for r in rows],device=model.device);mask=torch.tensor([[0]*(w-len(r))+[1]*len(r) for r in rows],device=model.device)
    done={}
    class Stop(StoppingCriteria):
     def __call__(self,input_ids,scores,**kw):
      flags=[]
      for i,r in enumerate(input_ids):
       s=tok.decode(r[w:].tolist(),skip_special_tokens=True).lstrip();yes='\n' in s
       if yes:done.setdefault(i,len(r)-w)
       flags.append(yes)
      return torch.tensor(flags,device=input_ids.device)
    with torch.inference_mode():res=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=80,do_sample=False,num_beams=1,eos_token_id=config['eos_ids'],pad_token_id=pad,stopping_criteria=StoppingCriteriaList([Stop()]))
    for i,(m,p,seq) in enumerate(zip(mails,prompts,res[:,w:].tolist())):
     if i in done:seq=seq[:done[i]]
     for j,v in enumerate(seq):
      if v in config['eos_ids']:seq=seq[:j+1];break
     raw=tok.decode(seq,skip_special_tokens=True)
     out.write(json.dumps({'style':style,'role':role,'id':m['id'],'prompt':p,'raw':raw,'answer':raw.strip().split('\n')[0],'token_ids':seq,'tokens':len(seq),'finish':'newline' if i in done else 'eos' if seq and seq[-1] in config['eos_ids'] else 'length','signature':signature},ensure_ascii=False)+'\n');count+=1
    save('status.json',{'phase':'running','completed':count,'target':32})
 save('status.json',{'phase':'complete','completed':count,'seconds':time.time()-start})
if __name__=='__main__':main()
