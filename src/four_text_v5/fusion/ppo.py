from dataclasses import dataclass
import math
import torch
from torch import nn
from .common import state,probabilities,convex,validate_matrix
from .learnability_audit import audit

class Policy(nn.Module):
    def __init__(self,hidden=32,state_mode="probabilities"):
        super().__init__()
        def net(n):return nn.Sequential(nn.Linear(4 if state_mode=="probabilities" else 8,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,n))
        self.actor,self.critic=net(4),net(1)
    def distribution(self,x):return torch.distributions.Dirichlet(torch.softmax(self.actor(x),dim=-1).clamp_min(1e-7)*4.)
    def weights(self,x):return self.distribution(x).mean

@dataclass(frozen=True)
class PPOConfig:
    state_mode:str="probabilities"
    seed:int=42
    refreshes:int=100
    epochs:int=4
    hidden:int=32
    lr:float=3e-4
    clip:float=.2
    entropy:float=.001
    false_negative_cost:float=1.
    false_positive_cost:float=1.
    threshold:float=.5
    actions_per_email:int=8

    def __post_init__(self):
        if self.state_mode not in ('probabilities','probabilities_entropy'):raise ValueError('invalid PPO state mode')
        if type(self.actions_per_email) is not int or self.actions_per_email<2:raise ValueError('at least two actions per email required')
        if any(not math.isfinite(v) for v in (self.lr,self.clip,self.entropy,self.false_negative_cost,self.false_positive_cost,self.threshold)):raise ValueError('finite PPO configuration required')
        if min(self.refreshes,self.epochs,self.hidden)<1 or self.lr<=0 or not 0<self.clip<1 or self.entropy<0 or min(self.false_negative_cost,self.false_positive_cost)<=0 or not 0<self.threshold<1:
            raise ValueError('invalid PPO configuration')

class PPOFusion:
    kind='ppo'
    def __init__(self,policy,source,signature,config,diagnostic,history,status='trained'):
        self.policy=policy.eval();self.source=source;self.calibrator_signature=signature
        self.config,self.diagnostic,self.history,self.status=config,diagnostic,history,status

    @classmethod
    def fit(cls,values,labels,source,signature,config=PPOConfig()):
        validate_matrix(values,labels)
        states=[state(p,config.state_mode) for p in values]
        diagnostic=audit(values,labels,config.threshold)
        torch.manual_seed(config.seed);policy=Policy(config.hidden,config.state_mode)
        if diagnostic['classification_changeable']==0:
            # Save an explicit non-learning artifact, not arbitrary trained weights.
            return cls(policy,source,signature,config,diagnostic,[],'not_learnable')
        x=torch.tensor(states);p=torch.tensor(values);y=torch.tensor(labels)
        optimizer=torch.optim.Adam(policy.parameters(),lr=config.lr);history=[]
        for refresh in range(config.refreshes):
            with torch.no_grad():
                dist=policy.distribution(x);actions=dist.sample((config.actions_per_email,));old=dist.log_prob(actions)
                fused=(actions*p).sum(-1)
                costs=torch.where(y==1,config.false_negative_cost,config.false_positive_cost)
                rewards=torch.where((fused>=config.threshold)==y.bool(),torch.ones_like(fused),-costs)
                # Leave-one-out baseline compares actions for the SAME email.
                # Other sampled actions are independent of this action; labels
                # stay confined to the fusion fitting split.
                baseline=(rewards.sum(0,keepdim=True)-rewards)/(config.actions_per_email-1)
                advantage=rewards-baseline
                advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
            for _ in range(config.epochs):
                dist=policy.distribution(x);ratio=(dist.log_prob(actions)-old).exp()
                actor=-torch.minimum(ratio*advantage,ratio.clamp(1-config.clip,1+config.clip)*advantage).mean()
                critic=torch.nn.functional.mse_loss(policy.critic(x).squeeze(-1),rewards.mean(0))
                loss=actor+.5*critic-config.entropy*dist.entropy().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite PPO loss')
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
            with torch.no_grad():
                prediction=(policy.weights(x)*p).sum(-1)
                predicted=prediction>=config.threshold
                history.append({'refresh':refresh+1,'brier':((prediction-y)**2).mean().item(),
                    'false_negatives':int(((~predicted)&(y==1)).sum()),'false_positives':int((predicted&(y==0)).sum())})
        return cls(policy,source,signature,config,diagnostic,history)

    def predict(self,scores):
        if self.status!='trained':raise ValueError('PPO has no learnable action space; upstream revision required')
        p=probabilities(scores,self.source,self.calibrator_signature)
        with torch.no_grad():w=self.policy.weights(torch.tensor(state(p,self.config.state_mode))).tolist()
        return {'probability':convex(p,w),'weights':w,'method':self.kind}

    def to_dict(self):
        return {'source':self.source,'calibrator_signature':self.calibrator_signature,'config':vars(self.config),
                'diagnostic':self.diagnostic,'history':self.history,'status':self.status,
                'state_dict':{k:v.detach().tolist() for k,v in self.policy.state_dict().items()}}

    @classmethod
    def from_dict(cls,value):
        cfg=PPOConfig(**value['config']);policy=Policy(cfg.hidden,cfg.state_mode)
        policy.load_state_dict({k:torch.tensor(v) for k,v in value['state_dict'].items()})
        if not all(torch.isfinite(p).all() for p in policy.parameters()):raise ValueError('invalid policy artifact')
        return cls(policy,value['source'],value['calibrator_signature'],cfg,value['diagnostic'],value['history'],value['status'])
