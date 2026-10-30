import copy
import json
import math
import tempfile
import unittest
from dataclasses import asdict,replace
from pathlib import Path
import torch
from four_text_v5.contracts import RawEmail,ROLES,TASKS
from four_text_v5.contracts.domain import DomainPolicy,task_for
from four_text_v5.contracts.evidence import strict_json,validate,schema
from four_text_v5.experts.catalog import feature_ids,load_role
from four_text_v5.ingestion import parse_email,InputBudget
from four_text_v5.ingestion.input_budget import slice_input
from four_text_v5.ingestion.identity import exclusion
from four_text_v5.scoring.role_scorer import RoleScorer,ScoringConfig
from four_text_v5.scoring.evidence import EvidenceAnalyzer,EvidenceConfig
from four_text_v5.scoring.render import render_classification,render_evidence
from four_text_v5.scoring.score_audit import audit_saved_scores
from four_text_v5.pipeline import Pipeline
from four_text_v5.cli import join_labels
from four_text_v5.fusion.common import state
from four_text_v5.fusion.ppo import PPOFusion,PPOConfig,Policy
from four_text_v5.training.split_manifest import TrainingRow,audit_splits
from four_text_v5.training.train_fusion import train_bundle,load_bundle,calibrated_rows
from four_text_v5.util import digest

torch.set_num_threads(1)

def mail(schema='enron_spam',body='Please reply with your preferred meeting time.',id='test'):
    return RawEmail(id,schema,schema,task_for(schema),'unit',body=body,recipient='Recipient')

class Tokenizer:
    def encode(self,text,add_special_tokens=False):return list(range(max(1,len(text.split()))))

class Backend:
    identity={'backend':'unit-test','experimental_results':False}
    max_tokens=18432
    tokenizer=Tokenizer()
    def __init__(self):self.calls=[]
    def score(self,prompt,candidates):
        self.calls.append(prompt)
        return dict(logprobs=[-4.,-2.],prompt_tokens=len(self.tokenizer.encode(prompt)),
            candidate_token_counts=[len(self.tokenizer.encode(' '+c)) for c in candidates])

class GenerationBackend(Backend):
    def __init__(self):super().__init__();self.generations=[]
    def generate_evidence(self,prompt,budget,contract,prefix):
        role=contract['properties']['agent_role']['enum'][0]
        ids=contract['properties']['features']['items']['properties']['feature_id']['enum']
        self.generations.append((role,list(ids)))
        # Valid ordinary evidence plus one invented quotation on the first attempt.
        first=len([r for r,_ in self.generations if r==role])==1
        features=[dict(feature_id=f,found=True,suspicious=False,evidence='invented absent quotation' if first and i==0 else 'Please reply') for i,f in enumerate(ids)]
        return dict(text=json.dumps(dict(agent_role=role,features=features)),finish_reason='stop',generated_tokens=20)

class V5Tests(unittest.TestCase):
    def test_default_joint_run_and_explicit_classification_only(self):
        backend=GenerationBackend();scorer=RoleScorer(backend)
        joint=Pipeline(scorer).run(mail(),True)
        self.assertIsNotNone(joint['evidence']);self.assertTrue(backend.generations)
        backend.generations.clear()
        only=Pipeline(scorer,evidence=None).run(mail(),True)
        self.assertIsNone(only['evidence']);self.assertFalse(backend.generations)
        self.assertEqual(joint['roles'],only['roles'])

    def test_classification_failure_does_not_skip_auxiliary_task(self):
        class FailingScore(GenerationBackend):
            def score(self,*args):raise ValueError('synthetic scoring failure')
        backend=FailingScore()
        record=Pipeline(RoleScorer(backend)).run(mail(),True)
        self.assertEqual(record['status'],'review_required')
        self.assertIn('scoring_failure',record['reason'])
        self.assertEqual(len(record['evidence']['records']),4)
        self.assertTrue(backend.generations)

    def test_native_labels_and_dual_prompts(self):
        expected={'action':4,'context':5,'persuasion':6,'purpose':4}
        for role,n in expected.items():
            self.assertEqual(len(feature_ids(role)),n)
            self.assertTrue(all(set(f)=={'feature_id','definition','selection_criteria','exclusion_rules'} for f in load_role(role)['features']))
        for label_schema in TASKS:
            m=parse_email(mail(label_schema));c=slice_input(m,0,len(m.body))
            for role in ROLES:
                p=render_classification(m,c,role);e,prefix=render_evidence(m,c,role)
                self.assertTrue(p.startswith('<|begin_of_text|>'));self.assertTrue(p.endswith('Classification:'))
                for label in TASKS[label_schema].candidates:self.assertIn(label,p)
                for fid in feature_ids(role):self.assertIn(fid,p);self.assertIn(fid,e)
                others=set().union(*(set(feature_ids(r)) for r in ROLES if r!=role))
                self.assertTrue(all('Feature ID: '+fid not in p for fid in others))
                self.assertTrue(e.endswith(prefix));self.assertNotIn('urgency_score',p+e)

    def test_score_without_generation_and_full_label_normalization(self):
        b=Backend();s=RoleScorer(b);scores,_=s.score(parse_email(mail('kaggle')))
        self.assertEqual(len(b.calls),8)
        # Both native labels have two tokens in this deliberately simple tokenizer.
        for r in scores:
            self.assertAlmostEqual(r.margin,1.)
            self.assertEqual(r.observations[0].summed_logprobs,(-4.,-2.))
        summed=RoleScorer(b,config=ScoringConfig('sum'))
        self.assertNotEqual(s.signature,summed.signature)
        self.assertAlmostEqual(summed.score(parse_email(mail('kaggle')))[0][0].margin,2.)

    def test_evidence_failure_never_changes_classification(self):
        class Broken:
            signature='broken'
            def analyze(self,m):raise RuntimeError('decoder failure')
        s=RoleScorer(Backend());a=Pipeline(s,evidence=None).run(mail(),True);b=Pipeline(s,evidence=Broken()).run(mail(),True)
        self.assertEqual(a['roles'],b['roles']);self.assertEqual(b['status'],'scored')
        self.assertEqual(a['decision_signature'],b['decision_signature']);self.assertNotEqual(a['run_signature'],b['run_signature'])
        self.assertEqual(b['evidence']['status'],'unreliable')

    def test_local_retry_preserves_good_fields(self):
        backend=GenerationBackend();analyzer=EvidenceAnalyzer(backend)
        result=analyzer.analyze(parse_email(mail()))
        for record in result['records']:
            self.assertEqual(record['status'],'valid');self.assertEqual(len(record['attempts']),2)
            self.assertEqual(record['attempts'][1]['feature_ids'],[feature_ids(record['role'])[0]])
            self.assertEqual(record['features'][1]['result']['evidence'],'Please reply')
        self.assertEqual(len(backend.generations),8)

    def test_failed_feature_not_replaced_with_false(self):
        result=EvidenceAnalyzer(GenerationBackend(),config=EvidenceConfig(max_retries=0)).analyze(parse_email(mail()))
        self.assertTrue(all(r['features'][0]['result'] is None for r in result['records']))
        self.assertTrue(all(r['status']=='partial' for r in result['records']))

    def test_quote_and_json_contract(self):
        with self.assertRaises(ValueError):strict_json('{"a":1,"a":2}')
        with self.assertRaises(ValueError):strict_json('{"a":NaN}')
        f=dict(feature_id='urgency',found=False,suspicious=True,evidence='')
        enc=lambda f:json.dumps(dict(agent_role='persuasion',features=[f]))
        good,bad=validate(enc(f),'persuasion',['urgency'],{'body':'reply now'})
        self.assertFalse(good);self.assertIn('urgency',bad)
        f.update(found=True,suspicious=False,evidence='reply  now')
        self.assertFalse(validate(enc(f),'persuasion',['urgency'],{'body':'reply now'})[0])
        good,_=validate(enc(f),'persuasion',['urgency'],{'body':'reply now'},'whitespace')
        self.assertEqual(good['urgency']['quote_match_mode'],'whitespace')

    def test_score_replay_and_join_reject_tampering(self):
        record=Pipeline(RoleScorer(Backend()),evidence=None).run(mail(),True)
        self.assertTrue(audit_saved_scores(record,parse_email(mail())))
        rows,excluded=join_labels([record],[dict(id='test',source_dataset='enron_spam',label=0)])
        self.assertEqual(len(rows),1);self.assertFalse(excluded);self.assertNotIn('evidence_quality',rows[0])
        broken=json.loads(json.dumps(record));broken['roles'][0]['observations'][0]['summed_logprobs'][0]=0.
        broken['record_sha256']=digest({k:v for k,v in broken.items() if k!='record_sha256'})
        with self.assertRaises(ValueError):join_labels([broken],[dict(id='test',source_dataset='enron_spam',label=0)])

    def test_state_dimensions_softmax_and_learning(self):
        self.assertEqual(len(state([.2,.3,.7,.8])),4)
        self.assertEqual(len(state([0.,1.,.5,.5],'probabilities_entropy')),8)
        self.assertEqual(state([0.,1.,.5,.5],'probabilities_entropy')[4:6],[0.,0.])
        with self.assertRaises(ValueError):state([.1,.2,.3,.4],'quality')
        policy=Policy();x=torch.zeros((2,4))
        self.assertTrue(torch.allclose(policy.weights(x),torch.softmax(policy.actor(x),-1),atol=1e-6))
        values=[[.1,.8,.2,.7],[.8,.1,.7,.2],[.2,.9,.1,.8],[.9,.2,.8,.1]]
        fitted=PPOFusion.fit(values,[0,1,0,1],'enron_spam','unit',PPOConfig(refreshes=2,epochs=1))
        self.assertEqual(fitted.status,'trained');self.assertEqual(len(fitted.history),2)
        restored=PPOFusion.from_dict(fitted.to_dict())
        self.assertTrue(torch.allclose(fitted.policy.weights(x),restored.policy.weights(x)))
        unlearnable=PPOFusion.fit([[.1,.2,.3,.4],[.6,.7,.8,.9]]*2,[0,1,0,1],'enron_spam','unit',PPOConfig(refreshes=1))
        self.assertEqual(unlearnable.status,'not_learnable')

    def test_mime_recipient_and_label_free_input(self):
        r=replace(mail(),body='',recipient='',mime='From: alice@example.com\nTo: bob@example.com\nSubject: Planning\n\nPlease reply tomorrow.')
        self.assertEqual(parse_email(r).recipient,'bob@example.com')
        with self.assertRaises(ValueError):RawEmail.from_dict(dict(asdict(mail()),label=0))

    def test_training_bundle_and_heldout_overlap(self):
        import random,string
        rng=random.Random(42)
        scorer=RoleScorer(Backend())
        def rows(prefix):
            result=[]
            for i in range(8):
                body=''.join(rng.choice(string.ascii_letters+' ') for _ in range(100))
                member=exclusion(parse_email(mail(id=prefix+str(i),body=body)))
                y=i%2;v=(-2 if y==0 else 2)
                result.append(TrainingRow(**asdict(member),label=y,margins=(v,-v,v*.5,-v*.5),scoring_signature=scorer.signature))
            return result
        calibration,fusion,validation=rows('c'),rows('f'),rows('v')
        heldout=[exclusion(parse_email(mail(id='heldout',body='This independent final input is held out.')))]
        policy=DomainPolicy('unit','spam','enron_spam',('enron_spam',))
        document=train_bundle(calibration,fusion,validation,heldout,policy,PPOConfig(refreshes=1,epochs=1))
        loaded,models,ppo,_=load_bundle(json.loads(json.dumps(document)))
        self.assertEqual(loaded['state_schema']['mode'],'probabilities')
        self.assertEqual(len(next(calibrated_rows(models,validation))),4)
        self.assertEqual(ppo.status,'trained')
        class Broken:
            signature='broken-evidence'
            def analyze(self,m):raise ValueError('invalid generated JSON')
        a=Pipeline(scorer,document,evidence=None).run(mail(id='new-prediction'))
        b=Pipeline(scorer,document,Broken()).run(mail(id='new-prediction'))
        self.assertEqual(a['status'],'ok');self.assertEqual(a['probability'],b['probability'])
        self.assertEqual(a['weights'],b['weights']);self.assertEqual(a['prediction'],b['prediction'])
        from four_text_v5.evaluation import evaluate
        evaluated=evaluate([a],[dict(id='new-prediction',source_dataset='enron_spam',label=0),dict(id='missing',source_dataset='enron_spam',label=1)])['enron_spam']
        self.assertEqual(evaluated['n'],2);self.assertEqual(evaluated['missing'],1)
        self.assertEqual(evaluated['coverage'],.5);self.assertIsNotNone(evaluated['ece10_successful_only'])
        old=copy.deepcopy(document);old['payload']['schema']='four-text-v4-model-reports-1';old['sha256']=digest(old['payload'])
        with self.assertRaises(ValueError):load_bundle(old)
        with self.assertRaises(ValueError):audit_splits(dict(a=calibration,b=calibration))

if __name__=='__main__':unittest.main()
