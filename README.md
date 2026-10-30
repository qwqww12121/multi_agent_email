# Multi-agent email analysis — V5

**Current version: Four-text V5, package 0.5.1.** Four agents analyze actions, context, persuasion and purpose using two independent prompts: native-label classification scoring and auxiliary structured evidence generation.

Classification consumes the email and role configuration directly. Evidence generation is a separate task; its JSON, length or quote failures do not determine whether classification can proceed. The current development default runs both tasks. Explicit `--no-evidence` selects classification only; `evidence` runs the auxiliary task independently.

## Roles and prompts

The four role configurations define 19 features, each with a definition, selection criteria and exclusion rules. Feature presence (`found`) and suspiciousness (`suspicious`) are distinct judgments. Evidence values are selected by the model; the code checks structure and original-text membership without rule-based semantic extraction. No demonstration emails or completed answers are inserted into production prompts.

- Runtime templates: [Classification](src/four_text_v5/prompts/common/classification_base.txt), [Evidence](src/four_text_v5/prompts/common/evidence_base.txt).
- Role configurations: [Action](src/four_text_v5/prompts/roles/action/role_definition.json), [Context](src/four_text_v5/prompts/roles/context/role_definition.json), [Persuasion](src/four_text_v5/prompts/roles/persuasion/role_definition.json), [Purpose](src/four_text_v5/prompts/roles/purpose/role_definition.json).
- Full review templates: [Action](design/rendered_prompts/action/), [Context](design/rendered_prompts/context/), [Persuasion](design/rendered_prompts/persuasion/), [Purpose](design/rendered_prompts/purpose/).

The classification model remains frozen Llama-3.1-8B Base. Candidate labels retain each dataset's native vocabulary and spam/phishing task meaning. Complete candidate sequences are scored; default scores are mean token log likelihoods, averaged over the two existing class-definition orders. Raw totals and token counts are retained. The sigmoid margin is an uncalibrated relative score.

The training implementation provides four monotone regularized Platt calibrators and clipped PPO with a four-probability state, or an explicitly configured probability-plus-entropy state. Evidence quality is not a PPO input. Changed scoring contracts reject incompatible old bundles. Details and limits are in [V5 protocol](docs/V5_PROTOCOL.md).

## Install and test

Python >=3.10. Supply an existing Llama-3.1-8B Base checkpoint matching the frozen file hashes; automatic model downloads are disabled.

```sh
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -e ".[model]"
python -m unittest discover -s tests -v
four-text-v5 --help
```

The 15 core tests cover independent branches, partial retries, native full-candidate scoring, quote/schema checks, split boundaries and synthetic calibration/PPO round trips. Optional transport tests exercise actual HF generation and LM Format Enforcer with synthetic logits. They do not establish real-model accuracy. Test fixtures never enter production prompts.

## Run

```sh
# Joint classification scoring and auxiliary evidence (current development default).
four-text-v5 score --input /data/emails.jsonl --output /runs/v5/joint.jsonl --model /models/Llama-3.1-8B
# Explicit classification-only task; it does not fit calibration or PPO.
four-text-v5 score --input /data/emails.jsonl --output /runs/v5/scores.jsonl --model /models/Llama-3.1-8B --no-evidence
# Independently scheduled auxiliary analysis.
four-text-v5 evidence --input /data/emails.jsonl --output /runs/v5/evidence.jsonl --model /models/Llama-3.1-8B
```

Inputs are label-free `RawEmail` JSONL records with `id`, `label_schema`, `source_dataset`, `task_type` and `calibration_domain`. Optional content fields include subject, sender, recipient, body, HTML or a sole MIME representation. See the [input contract](src/four_text_v5/contracts/email.py) and CLI help. All outputs require new paths. `predict` requires a separately trained, domain- and signature-matched V5 bundle; there is no silent uncalibrated fallback.

## Observed validation

The 2026-10-30 retrospective frozen four-source diagnostic processed 2,000 mails, 500 per source. Four-role scoring succeeded for 1,999; one mail exceeded the complete-input budget. Uncalibrated equal-mean accuracy at threshold 0.5 was **46.80%**. No calibration, PPO or RAG was run. Native task differences and the unscored sample remain explicit in the denominators.

The separate eight-mail auxiliary test produced 53 generation attempts: 28 strict JSON outputs, 66/152 fields passing final structure/quote checks, and 12/32 complete role reports passing those checks. Semantic spot checks still found sender/recipient confusion and irrelevant evidence. These results establish execution and expose limitations; they do not certify reliable role reasoning.

See [experiment notes](docs/V5_EXPERIMENT_NOTES.md), [aggregate metrics](qa/basic_metrics_20261030/metrics.json), and the [standalone metric view](qa/basic_metrics_20261030/basic_metrics_report.html). Evidence Accuracy and Role Consistency remain unestimated without complete semantic annotation; PPO improvement is unmeasured.

## Reproduce analysis with private artifacts

The frozen-sample runners accept the original run directories, verify their hashes and preserve sample IDs. They require separately supplied data and the original archive layout. The complete-candidate classifier and evidence budgets are recorded in the scripts; this repository does not include those private artifacts.

```sh
python scripts/calculate_basic_metrics.py --archive /private/results_20261030_final.tar.gz --output qa/basic_metrics_20261030/metrics.json
python scripts/build_metrics_view.py
node scripts/verify_metrics_view.cjs
```

The published metric JSON contains aggregate statistics only. The private archive, raw emails, per-mail labels/results, server connection details, model weights and local environments are excluded.

## Version provenance

V5 replaces the active R4 package in this repository. R4 remains in Git history at commit `cec5e526f76c4aa3ac603dad13369921ab1be944`; [R4 experiment notes](docs/R4_EXPERIMENT_NOTES.md) are historical. Historical six-column experiments, C3, and method-specific RAG banks remain separate.

[RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) records exact file hashes and the local source revision. Core V5 source, scripts, tests and runtime assets are copied byte-for-byte; newline conversion is disabled so code-based experiment signatures are preserved. Publication documentation and aggregate metric metadata are adapted for the repository.
