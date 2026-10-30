from dataclasses import asdict
from pathlib import Path
import torch
from ..util import digest,sigmoid
from ..contracts.domain import DomainPolicy
from ..contracts import ROLES,RawRoleScore,TASKS
from ..scoring.score_calibration import CalibratorSet
from ..fusion.ppo import PPOFusion,PPOConfig
from .train_calibrators import fit_calibrators,transform_rows
from .split_manifest import audit_splits,validate_training_rows
from .applicability import audit_calibration
from .applicability import auc

def implementation_signature():
    root=Path(__file__).resolve().parents[1]
    files=[p for folder in ('fusion','training') for p in (root/folder).glob('*.py')]
    files+=[root/'scoring/score_calibration.py',root/'pipeline.py',root/'contracts/domain.py']
    return digest({p.relative_to(root).as_posix():digest(p.read_bytes()) for p in sorted(files)})

def calibrated_rows(models,rows):
    for r in rows:
        yield tuple(models.apply(RawRoleScore(role,r.source,r.scoring_signature,TASKS[r.source].candidates,(),r.margins[i],sigmoid(r.margins[i]),0.,1.,
            r.source_dataset,r.task_type,r.calibration_domain,r.input_protocol,
            digest(dict(id=r.id,source_dataset=r.source_dataset,body_hash=r.body_hash)))) for i,role in enumerate(ROLES))

def train_bundle(calibration,fusion,validation,heldout,domain_policy,config=PPOConfig()):
    for rows in (calibration,fusion,validation):validate_training_rows(rows,domain_policy)
    if len({r.scoring_signature for r in calibration+fusion+validation})!=1:raise ValueError('split scoring signatures differ')
    if config.threshold!=domain_policy.threshold:raise ValueError('PPO and decision thresholds differ')
    splits=audit_splits(dict(calibration=calibration,fusion=fusion,validation=validation,heldout=heldout))
    models=fit_calibrators(calibration,domain_policy)
    # Development-only audit: expose instability without selecting experiments
    # by accuracy or looking at held-out labels.
    stage_auc={name:{role:auc([r.margins[i] for r in rows],[r.label for r in rows])
        for i,role in enumerate(ROLES)} for name,rows in
        (('calibration',calibration),('fusion',fusion),('validation',validation))}
    readiness=audit_calibration(models,validation)
    ppo=None;validation_report=None
    # Performance diagnostics never gate a scientific experiment. Numeric,
    # split/domain and learnability checks remain mandatory.
    if models is not None:
        values=transform_rows(models,fusion)
        ppo=PPOFusion.fit(values,[r.label for r in fusion],models.source,models.signature,config)
        if ppo.status!='trained':readiness['status']='review_required';readiness['issues'].append('PPO_not_learnable')
        else:
            probs=[ppo.predict(scores)['probability'] for scores in calibrated_rows(models,validation)]
            tp=tn=fp=fn=0
            for p,r in zip(probs,validation):
                pred=p>=domain_policy.threshold
                tp+=pred and r.label==1;tn+=not pred and r.label==0;fp+=pred and r.label==0;fn+=not pred and r.label==1
            validation_report=dict(n=len(validation),tp=tp,tn=tn,fp=fp,fn=fn,brier=sum((p-r.label)**2 for p,r in zip(probs,validation))/len(validation))
            validation_report['by_source']={}
            for source in domain_policy.source_datasets:
                cases=[(p,r.label) for p,r in zip(probs,validation) if r.source_dataset==source]
                cells={f'{name}_{label}':sum((p>=domain_policy.threshold)==label for p,y in cases if y==label) for label,name in ((0,'correct'),(1,'correct'))}
                support={label:sum(y==label for p,y in cases) for label in (0,1)}
                validation_report['by_source'][source]=dict(n=len(cases),negative=support[0],positive=support[1],tn=cells['correct_0'],tp=cells['correct_1'])
                for label in (0,1):
                    if support[label] and cells[f'correct_{label}']==0:
                        readiness['status']='review_required';readiness['issues'].append(f'PPO_misses_observed_source_class:{source}:{label}')
            if tp==0 or tn==0:
                readiness['status']='review_required';readiness['issues'].append('PPO_misses_entire_validation_class')
    payload=dict(schema='four-text-v5-dual-prompt-1',implementation_signature=implementation_signature(),domain_policy=asdict(domain_policy),scoring_signature=models.scoring_signature,
        calibrators=asdict(models),calibrator_signature=models.signature,ppo=ppo.to_dict() if ppo else None,
        readiness=readiness,validation_report=validation_report,split_audit=splits,ppo_config=asdict(config),
        fit_members=[r.member() for r in calibration+fusion],validation_members=[r.member() for r in validation],
        training_labels_signature=digest([asdict(r) for r in calibration+fusion]),
        validation_labels_signature=digest([asdict(r) for r in validation]),
        state_schema=dict(version='probabilities-5.0',roles=ROLES,mode=config.state_mode),
        development_role_auc=stage_auc,fixed_example_hashes=[],
        runtime=dict(torch=torch.__version__,cpu_threads=torch.get_num_threads()),
        limitations=['readiness is checked on development validation, not final-test efficacy',
                    'reports are generated by the model; exact quotations and JSON validity do not certify semantic correctness'])
    return dict(payload=payload,sha256=digest(payload))

def load_bundle(document):
    p=document['payload']
    if p.get('schema')!='four-text-v5-dual-prompt-1' or digest(p)!=document['sha256']:raise ValueError('bundle schema/hash mismatch; all pre-v5 bundles require retraining')
    if p.get('implementation_signature')!=implementation_signature():raise ValueError('calibration/PPO implementation changed; retrain')
    expected=dict(version='probabilities-5.0',roles=list(ROLES),mode=p['ppo_config']['state_mode'])
    actual=p.get('state_schema',{})
    if dict(actual,roles=list(actual.get('roles',())))!=expected:raise ValueError('state schema mismatch')
    policy=DomainPolicy.from_dict(p['domain_policy']);models=CalibratorSet.from_dict(p['calibrators'])
    if models.signature!=p['calibrator_signature'] or models.domain_policy!=policy or models.scoring_signature!=p['scoring_signature']:raise ValueError('calibration/domain binding mismatch')
    ppo=PPOFusion.from_dict(p['ppo']) if p['ppo'] else None
    config=PPOConfig(**p['ppo_config'])
    if config.threshold!=policy.threshold:raise ValueError('bundle threshold binding mismatch')
    if ppo and (ppo.source!=policy.label_schema or ppo.calibrator_signature!=models.signature or ppo.config!=config):raise ValueError('PPO binding mismatch')
    if p['readiness']['status']=='ready' and (ppo is None or ppo.status!='trained'):raise ValueError('ready bundle has no trained PPO')
    return p,models,ppo,policy
