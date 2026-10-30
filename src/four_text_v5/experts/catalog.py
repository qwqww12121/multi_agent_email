"""Load authored role definitions; never extract email features programmatically."""
import json
from importlib.resources import files
from ..contracts.task import ROLES


def resource(name):
    return files('four_text_v5.prompts').joinpath(name).read_text(encoding='utf-8')


def load_role(role):
    if role not in ROLES: raise ValueError('unknown role')
    value = json.loads(resource('roles/'+role+'/role_definition.json'))
    if value['agent_role'] != role: raise ValueError('role binding mismatch')
    ids = [f['feature_id'] for f in value['features']]
    if not ids or len(ids) != len(set(ids)): raise ValueError('unique role features required')
    for feature in value['features']:
        if set(feature) != {'feature_id','definition','selection_criteria','exclusion_rules'}:
            raise ValueError('complete feature definition required')
    return value


def feature_ids(role):
    return tuple(f['feature_id'] for f in load_role(role)['features'])


def render_catalog(role, selected=None):
    config = load_role(role)
    selected = feature_ids(role) if selected is None else tuple(selected)
    if not selected or len(selected) != len(set(selected)) or set(selected)-set(feature_ids(role)):
        raise ValueError('selected features must belong to this role')
    return '\n'.join('\n'.join((f"Feature ID: {f['feature_id']}",
        'Definition: '+f['definition'], 'Selection criteria: '+f['selection_criteria'],
        'Exclusion rules: '+f['exclusion_rules'])) for f in config['features'] if f['feature_id'] in selected)
