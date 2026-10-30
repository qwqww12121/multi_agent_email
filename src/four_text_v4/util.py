import dataclasses
import hashlib
import json
import math
from pathlib import Path

def primitive(value):
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value

def digest(value):
    b = value if isinstance(value, bytes) else json.dumps(primitive(value), sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf8')
    return hashlib.sha256(b).hexdigest()

def sigmoid(value):
    if not math.isfinite(value): raise ValueError('nonfinite margin')
    return 1/(1+math.exp(-value)) if value>=0 else math.exp(value)/(1+math.exp(value))

def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(s) for s in f if s.strip()]

def save_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(primitive(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)
