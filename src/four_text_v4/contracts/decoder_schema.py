"""Decoder projection; cross-field if/then rules stay in full post-validation."""
from copy import deepcopy
from .json_schema import report_schema


def decoder_schema(role):
    source = report_schema(role)
    defs = source['$defs']

    def expand(node):
        if isinstance(node, list):
            return [expand(x) for x in node]
        if not isinstance(node, dict):
            return node
        node = deepcopy(node)
        if '$ref' in node:
            ref = node.pop('$ref')
            if not ref.startswith('#/$defs/'):
                raise ValueError('unsupported schema reference')
            base = deepcopy(defs[ref[len('#/$defs/'):]])
            base['allOf'] = base.get('allOf', []) + node.pop('allOf', [])
            base.update(node)
            node = base
        for condition in node.pop('allOf', []):
            if 'if' in condition:
                continue  # documented post-generation obligation
            if set(condition) != {'properties'}:
                raise ValueError('unsupported schema intersection')
            for key, constraints in condition['properties'].items():
                node['properties'][key].update(constraints)
        for key in ('$defs', '$schema', 'description'):
            node.pop(key, None)
        if 'const' in node:
            node['enum'] = [node.pop('const')]
        return {key: expand(value) for key, value in node.items()}

    result = expand(source)
    main = 'requested_actions' if role == 'action' else 'influence_attempts' if role == 'persuasion' else None
    keys = ['schema_version', 'role']
    if main:
        keys.append(main)
    keys += [key for key in result['properties'] if key not in keys and key != 'uncertainties']
    keys.append('uncertainties')
    result['properties'] = {key: result['properties'][key] for key in keys}
    result['required'] = keys
    return result


def validate_full_schema(report, role):
    from jsonschema import Draft202012Validator, ValidationError
    try:
        Draft202012Validator(report_schema(role)).validate(report)
    except ValidationError as exc:
        raise ValueError("schema: " + exc.json_path + ": " + exc.message) from exc
