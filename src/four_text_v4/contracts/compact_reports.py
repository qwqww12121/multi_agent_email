"""Lossless wire representation; no extraction, defaults or field repair."""
VERSION = 'model-role-fields-4.1-compact'

def compact(value):
    if isinstance(value,dict):
        if set(value)=={'value','status','scope','evidence'}:
            return [value['value'],value['status'],value['scope'],[[c['field'],c['quote']] for c in value['evidence']]]
        return {k:VERSION if k=='schema_version' else compact(v) for k,v in value.items()}
    if isinstance(value,list):return [compact(v) for v in value]
    return value

def expand(report,role):
    from .reports import SINGLE_FIELDS,LIST_FIELDS,ACTION_FIELDS,INFLUENCE_FIELDS,shape,REPORT_VERSION
    if not isinstance(report,dict) or set(report)!=set(shape(role)):
        raise ValueError('compact report requires exact top-level fields')
    if report['schema_version']!=VERSION or report['role']!=role:raise ValueError('compact version/role mismatch')
    def claim(value):
        if not isinstance(value,list) or len(value)!=4 or not isinstance(value[3],list):
            raise ValueError('claim requires [value,status,scope,[[source,quote],...]]')
        if any(not isinstance(c,list) or len(c)!=2 for c in value[3]):raise ValueError('citation requires [source,quote]')
        return dict(value=value[0],status=value[1],scope=value[2],evidence=[dict(field=c[0],quote=c[1]) for c in value[3]])
    result=dict(schema_version=REPORT_VERSION,role=role)
    for key in (*SINGLE_FIELDS[role],'uncertainties'):result[key]=claim(report[key])
    for key in LIST_FIELDS[role]:
        if not isinstance(report[key],list):raise ValueError('claim list required')
        result[key]=[claim(item) for item in report[key]]
    for key,fields in (('requested_actions',ACTION_FIELDS),('influence_attempts',INFLUENCE_FIELDS)):
        if key not in report:continue
        if not isinstance(report[key],list):raise ValueError('record list required')
        result[key]=[]
        for item in report[key]:
            if not isinstance(item,dict) or set(item)!=set(fields):raise ValueError('record requires exact fields: '+','.join(fields))
            result[key].append({k:claim(item[k]) for k in fields})
    return result
