# V5 protocol and implementation boundaries

V5 implements role definitions + two prompts + four independent native-label scores + per-role calibration + PPO fusion. The deployed diagnostic used the scoring and auxiliary branches; it did not fit calibration or PPO.

## Roles

- Action: recipient_action, sensitive_action, abnormal_instruction, action_consequence.
- Context: sender_identity, recipient_identity, organization_identity, relationship_context, contextual_consistency.
- Persuasion: urgency, reward, fear, authority, emotional_appeal, flattery.
- Purpose: primary_purpose, commercial_promotion, deceptive_intent, content_consistency.

Each feature has a definition, selection criteria and exclusion rules. Action consequences include context-supported potential risk, not just explicitly stated immediate harm. A possible risk is not a verified outcome. Shared role definitions specify analysis responsibilities; the two task templates supply their distinct output instructions.

## Classification

The Base completion prompt contains dataset/task definitions, role configuration, complete input and a fixed `Classification:` prefix. The program scores every token of each native candidate, including a fixed leading space. Default normalization divides summed log likelihood by candidate token count. Positive-minus-negative margins are averaged over negative-first and positive-first definition orders; complete chunks, if explicitly enabled, are weighted by body characters. `sigmoid(margin)` is an uncalibrated relative score.

Native labels: CEAS/ePhishLLM 0/1, Enron ham/spam, Kaggle Safe Email/Phishing Email, Nazario/PhishingPot Legit/Phishing. Generic advertising is not automatically phishing. Consent, identity and unseen attachment content are not invented.

Default input budget: 18,432 tokens, reject complete over-budget inputs without truncation. Explicit `chunk_all` changes the signed input protocol. Model, code, role assets, prompt, budget, normalization and runtime versions are bound into signatures.

## Evidence

Each role returns agent_role and features; each feature returns feature_id, found, suspicious and evidence. A present feature may be legitimate. found=false requires suspicious=false and empty evidence. A found feature requires an exact original passage. Failed extraction remains unavailable, not automatically false.

Evidence is an independently schedulable auxiliary task. The current development default is joint execution, with explicit --no-evidence for classification only. Joint execution still attempts scheduled evidence after a classification result fails. Evidence outcomes do not alter the primary decision.

The current transport uses Base BOS, a structural role prefix and LM Format Enforcer 0.11.3. The post-validator checks role, complete unique feature membership, types, cross-field requirements and source quotes. Evidence generation allows 1,024 new tokens, one retry by default (bounded 0–2), and 2,000 characters per quote. Only failed fields are retried. Greedy generation and all raw attempts are retained; no semantic output repair occurs. Optional whitespace matching is signed and explicitly recorded.

These token, quote and retry limits, two-order scoring and decoding implementation are engineering choices, not values prescribed by the original design. Valid JSON or exact string membership does not establish semantic accuracy. A full all-false report can pass format validation without contributing any nonempty evidence.

## Calibration and PPO

The implemented calibrator is a positive-slope regularized Platt form on each role margin. Calibration, PPO fitting, validation and final-test exclusion manifests are separate and checked by source ID, canonical text hashes, known families, alternate text views and cross-split near duplicates. Test labels never fit a calibrator or policy.

PPO state is four calibrated probabilities, or four probabilities plus binary entropies when explicitly configured. It has no evidence-quality inputs. Actor softmax controls simplex weights; the retained PPO explores with Dirichlet concentration 4*softmax(actor), with a numerical lower bound, and uses distribution means for inference. Correct classifications receive +1; false negatives and false positives receive separately frozen negative costs. Costs and threshold must be selected before final testing.

If all possible convex combinations remain on the same side of the threshold for every fitting row, the discrete classification reward cannot distinguish actions; the implementation records not_learnable and does not claim trained fusion. This guard, monotone calibration and strict artifact compatibility are additional engineering policies. They require empirical validation and are not evidence that all future calibration/PPO choices will fail.

## Scope remaining

Core implementation and synthetic software verification are complete. Real evidence semantics remain unreliable in spot checks. Temperature/Isotonic calibration comparisons, the full ablation orchestration, complete semantic annotation and formal trained-PPO comparisons remain future experiments. No result from an old signature is presented as a V5 result.
