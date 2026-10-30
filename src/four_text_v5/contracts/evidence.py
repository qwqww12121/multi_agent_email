"""Validate syntax and source membership, never infer semantic features by rules."""
import json
import re
from ..experts.catalog import feature_ids

VERSION='role-features-5.0'

def strict_json(text):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate JSON key: '+k)
            result[k]=v
        return result
    def nonfinite(value):raise ValueError('nonfinite JSON: '+value)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=nonfinite)

def schema(role,selected=None):
    ids=list(feature_ids(role) if selected is None else selected)
    if not ids or len(set(ids))!=len(ids) or not set(ids)<=set(feature_ids(role)):raise ValueError('invalid feature subset')
    return dict(type='object',additionalProperties=False,required=['agent_role','features'],properties=dict(
        agent_role=dict(type='string',enum=[role]),features=dict(type='array',minItems=len(ids),maxItems=len(ids),
        items=dict(type='object',additionalProperties=False,required=['feature_id','found','suspicious','evidence'],properties=dict(
            feature_id=dict(type='string',enum=ids),found=dict(type='boolean'),suspicious=dict(type='boolean'),
            evidence=dict(type='string',maxLength=2000))))))

def validate(text,role,selected,sources,quote_mode='exact'):
    if quote_mode not in ('exact','whitespace'):raise ValueError('invalid quote mode')
    ids=tuple(selected);good={};errors={}
    try:
        value=strict_json(text)
        if type(value) is not dict or set(value)!= {'agent_role','features'} or value['agent_role']!=role or type(value['features']) is not list:
            raise ValueError('invalid role/envelope')
        items=value['features']
        if any(type(f) is not dict or f.get('feature_id') not in ids for f in items):raise ValueError('unknown feature or non-object entry')
    except (ValueError,TypeError) as exc:
        return {},{i:str(exc) for i in ids}
    for fid in ids:
        matches=[f for f in items if f['feature_id']==fid]
        try:
            if len(matches)!=1:raise ValueError('feature missing or duplicated')
            f=matches[0]
            if set(f)!= {'feature_id','found','suspicious','evidence'}:raise ValueError('unexpected feature keys')
            if type(f['found']) is not bool or type(f['suspicious']) is not bool or type(f['evidence']) is not str:raise ValueError('invalid field types')
            quote=f['evidence'];matched=[];mode='none'
            if len(quote)>2000:raise ValueError('evidence too long')
            if not f['found']:
                if f['suspicious'] or quote!='':raise ValueError('absent feature requires false/empty evidence')
            else:
                if not quote.strip():raise ValueError('found feature requires evidence')
                matched=[k for k,v in sources.items() if quote in v];mode='exact'
                if not matched and quote_mode=='whitespace':
                    norm=lambda s:re.sub(r'\s+',' ',s).strip()
                    matched=[k for k,v in sources.items() if norm(quote) in norm(v)];mode='whitespace'
                if not matched:raise ValueError('quote not present in supplied source fields')
            good[fid]=dict(result=f,quote_match_mode=mode,matched_source_fields=matched)
        except (ValueError,TypeError) as exc:errors[fid]=str(exc)
    return good,errors
