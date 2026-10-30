from collections import Counter
import math
from ..util import digest

def auc(values,labels):
    pos=[v for v,y in zip(values,labels) if y];neg=[v for v,y in zip(values,labels) if not y]
    return sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg)) if pos and neg else None

def audit_calibration(models,validation):
    policy=models.domain_policy;labels=[r.label for r in validation];issues=[];sources={}
    for source in policy.source_datasets:
        counts=Counter(r.label for r in validation if r.source_dataset==source)
        sources[source]={'negative':counts[0],'positive':counts[1]}
        if not sum(counts.values()):issues.append('source_not_validated:'+source)
        if policy.validation_scope=='within_source' and min(counts[0],counts[1])<policy.min_class_count:issues.append('insufficient_source_class_support:'+source)
    counts=Counter(labels)
    if min(counts[0],counts[1])<policy.min_class_count:issues.append('insufficient_domain_class_support')
    reports=[]
    for i,model in enumerate(models.models):
        margins=[r.margins[i] for r in validation];values=[model.probability(v) for v in margins]
        ranking=auc(margins,labels);spread=max(values)-min(values);reasons=[]
        if ranking is None or ranking<=policy.min_auc:reasons.append('no_validated_positive_ranking')
        if model.slope<policy.min_slope:reasons.append('collapsed_calibration_slope')
        if spread<policy.min_probability_spread:reasons.append('near_constant_probabilities')
        source_auc={s:auc([r.margins[i] for r in validation if r.source_dataset==s],[r.label for r in validation if r.source_dataset==s]) for s in sources}
        if policy.validation_scope=='within_source' and any(a is None or a<=policy.min_auc for a in source_auc.values()):reasons.append('source_ranking_not_supported')
        threshold_margin=model.center+(math.log(policy.threshold/(1-policy.threshold))-model.intercept)*model.scale/model.slope
        reports.append(dict(role=model.role,auc=ranking,source_auc=source_auc,probability_range=[min(values),max(values)],
            slope=model.slope,decision_margin=threshold_margin,positive_decisions=sum(p>=policy.threshold for p in values),issues=reasons))
    # Specialists need not all be strong standalone classifiers: PPO retains
    # its job of learning conditional weights. Individual weaknesses are warnings.
    warnings=[r['role']+':'+reason for r in reports for reason in r['issues']]
    if all(r['issues'] for r in reports):issues.append('all_roles_lack_validated_calibration_signal')
    return dict(status='ready' if not issues else 'review_required',issues=issues,warnings=warnings,roles=reports,source_class_support=sources,
        validation_scope=policy.validation_scope,validation_signature=digest([r.member() for r in validation]),
        limitation='Declared transfer validation is not same-source two-class validation; readiness is not a final-test efficacy claim.')
