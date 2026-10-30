"""Export readable role templates and separate decoder/validator schemas; no examples."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from four_text_v4.contracts import ROLES
from four_text_v4.contracts.json_schema import report_schema
from four_text_v4.contracts.decoder_schema import decoder_schema
from four_text_v4.experts.cards import role_card, PROMPT_VERSION
from four_text_v4.experts.base_prompt import BOS, ANSWER_MARKER, report_prefix
folder = ROOT/'prompts'
folder.mkdir(exist_ok=True)
for role in ROLES:
    text = (BOS + 'EMAIL SPECIALIST REPORT TASK\n<DATASET_TASK_DEFINITION>\n'
            + role_card(role) + '\n<INPUT_SCOPE_AND_EMAIL_DATA_JSON>\nEnd email data.\n'
            + 'Complete only the ' + role + ' report.' + ANSWER_MARKER + report_prefix(role))
    (folder/(role+'.txt')).write_text(text, encoding='utf-8')
    for kind, schema in [('validation',report_schema(role)), ('decoder',decoder_schema(role))]:
        (folder/(role+'.'+kind+'.schema.json')).write_text(json.dumps(schema, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(prompt_version=PROMPT_VERSION, roles=list(ROLES), demonstrations=0,
                     enforcement='lm-format-enforcer token mask; full schema and source checks after generation')))
