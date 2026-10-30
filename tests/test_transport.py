"""Synthetic logits test transport arithmetic and constraints, not Llama efficacy."""
import importlib.util
import json
import math
import string
import types
import unittest
import torch
from four_text_v5.contracts.evidence import schema
from four_text_v5.scoring.candidate_backend import score_model
from four_text_v5.scoring.json_transport import seeded_parser,generate_json,object_end

class CandidateTests(unittest.TestCase):
    def test_all_tokens_are_conditionally_scored(self):
        class Tokenizer:
            def __call__(self,text,add_special_tokens=False):
                return {'input_ids':{'prefix':[0,1],' ham':[2],' phishing email':[3,4]}[text]}
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.inputs=[]
            def forward(self,ids,use_cache=False):
                self.inputs.append(ids.tolist()[0])
                return types.SimpleNamespace(logits=torch.zeros((1,ids.shape[1],5)))
        model=Model();answer=score_model(model,Tokenizer(),'prefix',('ham','phishing email'),64)
        self.assertEqual(model.inputs,[[0,1,2],[0,1,3,4]])
        self.assertEqual(answer['candidate_token_counts'],[1,2])
        self.assertAlmostEqual(answer['logprobs'][0],-math.log(5),places=6)
        self.assertAlmostEqual(answer['logprobs'][1],-2*math.log(5),places=6)

@unittest.skipUnless(importlib.util.find_spec('lmformatenforcer'),'optional model dependencies unavailable')
class DecoderTests(unittest.TestCase):
    def test_schema_and_quote_characters(self):
        from four_text_v5.experts.catalog import feature_ids
        for role in ('action','context','purpose','persuasion'):
            text=json.dumps(dict(agent_role=role,features=[dict(feature_id=i,found=False,suspicious=False,evidence='') for i in feature_ids(role)]),separators=(',',':'))
            p=seeded_parser(schema(role),'')
            for char in text:
                self.assertIn(char,p.get_allowed_characters());p=p.add_character(char)
            self.assertTrue(p.can_end())
        self.assertIsNone(object_end('{"evidence":"}'))
        self.assertEqual(object_end('{"evidence":"}"} tail'),16)

    def test_actual_hf_generation_masks_invalid_token(self):
        from tokenizers import Tokenizer,models,pre_tokenizers,decoders
        from transformers import GPT2Config,GPT2LMHeadModel,PreTrainedTokenizerFast
        from transformers.modeling_outputs import CausalLMOutputWithPast
        bos='<|begin_of_text|>'
        symbols=[bos,'<eos>','<unk>','<pad>']+list(string.printable)
        core=Tokenizer(models.WordLevel({c:i for i,c in enumerate(symbols)},unk_token='<unk>'))
        core.pre_tokenizer=pre_tokenizers.Split('',behavior='isolated');core.decoder=decoders.Fuse()
        tok=PreTrainedTokenizerFast(tokenizer_object=core,bos_token=bos,eos_token='<eos>',unk_token='<unk>',pad_token='<pad>')
        model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tok),n_embd=8,n_layer=1,n_head=1,n_positions=2048,
            eos_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,pad_token_id=tok.pad_token_id)).eval()
        prefix='{"agent_role":"action","features":'
        target=prefix+'[{"feature_id":"recipient_action","found":true,"suspicious":false,"evidence":"reply"}]}'
        tokens=tok.encode(target[len(prefix):],add_special_tokens=False);calls=[0]
        def forward(this,input_ids=None,attention_mask=None,**kwargs):
            step=calls[0];calls[0]+=1
            logits=torch.zeros((1,1,len(tok)));logits[0,0,tok.unk_token_id]=100.
            logits[0,0,tokens[step] if step<len(tokens) else tok.eos_token_id]=20.
            return CausalLMOutputWithPast(logits=logits)
        model.forward=types.MethodType(forward,model)
        backend=types.SimpleNamespace(tokenizer=tok,model=model,max_tokens=2048)
        answer=generate_json(backend,bos+'Evidence JSON:\n'+prefix,len(tokens)+2,schema('action',['recipient_action']),prefix)
        self.assertEqual(answer['text'],target);self.assertEqual(answer['raw_text'],target[len(prefix):])
        self.assertEqual(answer['finish_reason'],'stop');self.assertEqual(answer['stop_cause'],'complete_json_object')

if __name__=='__main__':unittest.main()
