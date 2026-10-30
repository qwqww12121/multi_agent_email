# Version provenance

User selected four_agent_failure_case.html on 2026-10-30. The HTML identifies R5, chat template plus JSON constraint, ephishllm:002649. Matching arm: chat_json in diagnostics/r5/probe.py. All four arms are preserved. Core source: four_text_v4r4, the implementation imported by historical R5.

Only import, model and output paths are made portable using environment variables. Task strings, schema, prefill, budgets and stopping conditions are unchanged. Modified script hashes produce a NEW run signature; this export is not the historical run.

The original HTML embeds dataset text and model outputs and remains local. No raw email data is republished. Optional short_probe.py QA/cloze does not replace chat_json.

Historical chat_json: 16/16 schema valid, 6/16 nonempty verbatim evidence. Verbatim does not imply role relevance. Classification, full reports and PPO were not accepted by this diagnostic. R4 documentation contains historical paths and process snapshots.
