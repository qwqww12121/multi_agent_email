# Multi-agent email analysis - R4

**Selected version: four_text_v4r4, package 0.4.4.** Four roles (action, context, persuasion, purpose) generate full structured reports using Llama-3.1-8B Base completion prompts and constrained JSON decoding.

The active repository contains R4 prompts and code only. R5 minimal-evidence prompts and diagnostic runners are removed from the current tree. Earlier commits remain in Git history.

## Prompt and output protocol

Base start token -> dataset task definition -> shared evidence requirements -> role duties and field definitions -> full email -> explicit report prefix. No demonstration emails or completed example reports are inserted. Reports retain named value/status/scope/evidence fields. Generation uses LM Format Enforcer token filtering, followed by full schema and original-quote validation. Structural validity does not establish semantic correctness.

Read the exact exported role templates:

- [Action](prompts/action.txt)
- [Context](prompts/context.txt)
- [Persuasion](prompts/persuasion.txt)
- [Purpose](prompts/purpose.txt)

Decoder and full validation schemas sit beside each template. Actual requests are assembled by src/four_text_v4/scoring/render.py. Core source, scripts, tests and prompt exports are copied byte-for-byte from the local R4 source.

## Install and test

Python >=3.11. Use a local, lawfully obtained Llama-3.1-8B Base checkpoint matching the frozen model-file hashes. The loader does not download or replace model weights.

```sh
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -e ".[model]"
python -m unittest discover -s tests -v
four-text-v4 --help
```

## Score emails with R4 reports

Input is JSONL containing RawEmail objects; see src/four_text_v4/contracts/email.py and the CLI help for supported fields and task/domain bindings. The command below explicitly uses the historical R4 1536-token report allowance and 18432-token context budget. The generic CLI default report allowance is 4096; do not omit the explicit 1536 value when matching the diagnostic budget.

```sh
four-text-v4 score --input /data/emails.jsonl --output /runs/r4/scores.jsonl --model /models/Llama-3.1-8B --report-max-new-tokens 1536 --max-tokens 18432
```

The score command generates reports and computes dataset-native candidate-label likelihoods; it does not fit calibration or PPO. Full training/prediction commands exist in the core package and require separately fitted, protocol-matched artifacts. Old scores or calibration/PPO artifacts must not be reused across changed scoring signatures.

The historical four-dataset batch diagnostic is scripts/reference_runner.py. It retains the original remote directory layout and frozen-input reuse assumptions. See [R4 experiment notes](docs/R4_EXPERIMENT_NOTES.md) for the exact reuse-sample and run commands, historical runtime settings and limitations.

## Validation boundaries

Historical real-Base probe: four emails, 16 role reports; six parseable JSON outputs, ten token-limit failures, zero reports passing the full schema and quote checks. No valid final classification result was established. These are historical diagnostic findings, not new measurements or a classification accuracy of zero. The probe did not train calibration, PPO or use RAG.

The repository contains research implementations of native-label scoring, calibration and PPO, but this R4 diagnostic does not establish their real-data effectiveness. Historical six-column experiments, C3 and method-specific RAG banks remain separate.

## Publication contents

Source, exact prompts/schemas, tests, dependency configuration, provenance and SHA-256 inventory are included. Datasets, raw emails, credentials, weights, historical model outputs and local QA dependencies are excluded. The local HTML containing a full real-email example is not republished. No source-code license has been selected; model and dataset licenses remain the user's responsibility.
