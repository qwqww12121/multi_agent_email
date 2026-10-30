from dataclasses import asdict
from .contracts import RawEmail,TASKS
from .ingestion import parse_email,BudgetExceeded
from .ingestion.identity import exclusion
from .scoring.role_scorer import RoleScoringFailure
from .training.train_fusion import load_bundle
from .training.split_manifest import assert_unseen
from .util import digest

_AUTO_EVIDENCE=object()

class Pipeline:
    def __init__(self,scorer,bundle=None,evidence=_AUTO_EVIDENCE):
        if evidence is _AUTO_EVIDENCE:
            from .scoring.evidence import EvidenceAnalyzer
            evidence=EvidenceAnalyzer(scorer.backend,scorer.budget)
        self.scorer,self.evidence=scorer,evidence
        self.bundle,self.models,self.ppo,self.policy=load_bundle(bundle) if bundle else (None,None,None,None)
        if self.bundle and self.bundle['scoring_signature']!=scorer.signature:raise ValueError('classification contract changed; retrain calibration/PPO')
        self.decision_signature=digest(dict(scoring=scorer.signature,bundle=digest(bundle) if bundle else None,method='dual-prompt-ppo-v5'))
        self.run_signature=digest(dict(decision=self.decision_signature,evidence=evidence.signature if evidence else None))

    def run(self,raw:RawEmail,score_only=False):
        mail=parse_email(raw);member=exclusion(mail)
        result=dict(id=raw.id,source=raw.source,source_dataset=raw.source_dataset,task_type=raw.task_type,
            calibration_domain=raw.calibration_domain,input_protocol=raw.input_protocol,body_hash=mail.body_hash,
            family_id=member.family_id,content_hashes=member.content_hashes,status='review_required',prediction=None,
            native_prediction=None,probability=None,reason=mail.fatal,run_signature=self.run_signature,
            decision_signature=self.decision_signature,scoring_signature=self.scorer.signature,
            experimental_results=self.scorer.backend.identity.get('experimental_results',True),
            input=asdict(mail),roles=[],method='candidate_scores' if score_only else 'ppo',
            scoring_config=asdict(self.scorer.config),input_budget=asdict(self.scorer.budget),evidence=None)
        def finish():
            # Auxiliary analysis is independent even when primary scoring fails.
            if self.evidence:
                try:result['evidence']=self.evidence.analyze(mail)
                except Exception as exc:
                    result['evidence']=dict(signature=self.evidence.signature,status='unreliable',error=f'{type(exc).__name__}: {exc}')
            result['record_sha256']=digest(result)
            return result
        if mail.fatal:return finish()
        if not score_only:
            if not self.bundle:raise ValueError('trained domain-bound PPO bundle required; no uncalibrated fallback')
            if not self.policy.accepts(raw):result['reason']='unsupported_task_source_or_calibration_domain';return finish()
            try:assert_unseen(member.member(),self.bundle['fit_members']+self.bundle['validation_members'])
            except ValueError as exc:result['reason']=str(exc);return finish()
            result['development_diagnostics']=self.bundle['readiness']
            if self.ppo is None or self.ppo.status!='trained':result['reason']='trained_PPO_required';return finish()
        try:scores,chunks=self.scorer.score(mail)
        except BudgetExceeded as exc:result['reason']='input_budget: '+str(exc);return finish()
        except RoleScoringFailure as exc:
            result.update(reason='scoring_failure: '+str(exc),roles=[asdict(s) for s in exc.completed],partial_observations=[asdict(s) for s in exc.partial]);return finish()
        result.update(roles=[asdict(s) for s in scores],coverage=dict(complete=True,chunks=[asdict(c) for c in chunks]),reason=None)
        if score_only:result['status']='scored'
        else:
            calibrated=tuple(self.models.apply(s) for s in scores);decision=self.ppo.predict(calibrated)
            pred=int(decision['probability']>=self.policy.threshold)
            result.update(status='ok',prediction=pred,native_prediction=TASKS[raw.source].candidates[pred],
                probability=decision['probability'],weights=decision['weights'],calibrated_roles=[asdict(s) for s in calibrated])
        return finish()
