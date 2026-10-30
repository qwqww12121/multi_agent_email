# Multi-agent email analysis

Four-role email analysis research code, selected by the user from **four_agent_failure_case.html**: **R5 / chat_json**, with the complete four-arm diagnostic and its actual R4 core dependency.

Roles: action, purpose, persuasion, context. R5 asks each role to generate a single evidence string of at most 160 characters. R5 does not run classification, calibration, PPO or RAG. The core package retains the research implementations for full reports, native candidate-label likelihood scoring, calibration and PPO.

## Status

This is a diagnostic research version with known semantic failures. Historical chat_json: 16/16 schema valid, 6/16 nonempty verbatim evidence. Exact quotation does not establish role relevance. Full reports, classification and PPO are not validated by this probe. See [version provenance](docs/VERSION.md), [R5 diagnostic notes](diagnostics/r5/README.md) and [R4 historical notes](docs/R4_HISTORY.md). Historical six-column experiments and C3 remain separate.

## Install and check

Python >=3.11. Real inference requires sufficient GPU memory and a lawfully obtained local Llama-3.1-8B Base checkpoint. The loader enforces the historical frozen file hashes and never downloads or substitutes weights.

```sh
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -e ".[model]"
python -m unittest discover -s tests -v
```

## Run the selected R5 diagnostic

Create a fresh existing run directory containing inputs.jsonl: one JSON object per line with string fields id and body. Include any visible subject in body. Historical diagnostics used four emails; new inputs create a new experiment and do not reproduce historical results. Do not reuse a directory containing previous outputs.

PowerShell:

```powershell
$env:LLAMA_MODEL_PATH = 'D:/models/Llama-3.1-8B'
$env:MAIL_AGENT_RUN_DIR = 'D:/runs/r5-new'
python diagnostics/r5/probe.py
```

Linux:

```sh
export LLAMA_MODEL_PATH=/models/Llama-3.1-8B
export MAIL_AGENT_RUN_DIR=/runs/r5-new
python diagnostics/r5/probe.py
```

Four roles and four arms run per email. Filter records.jsonl by arm=chat_json for the selected HTML protocol. Full prompts, raw generations, tokens, stop reasons and signatures are retained without repairing failed outputs.

Optional later QA/cloze probe: `python diagnostics/r5/short_probe.py`. It uses a different output protocol. Only after both probes complete, run `python diagnostics/r5/audit.py`; this historical auditor requires four inputs, 64 main records and 32 follow-up records.

## Layout

- src/four_text_v4: R4 core, contracts, scoring and model identity checks.
- diagnostics/r5: selected R5 prompts, runner and auditor.
- scripts: R4 full-report utilities; reference_runner retains historical remote layout assumptions. Use the portable R5 entry above.
- tests: report structure, quotes, model identity and generation constraints.
- docs: version provenance and historical status.
- RELEASE_MANIFEST.json: file SHA-256 inventory.

Datasets, raw emails, credentials, weights, caches and run outputs are excluded. Users must satisfy model and dataset license requirements. No source-code license has been selected.
