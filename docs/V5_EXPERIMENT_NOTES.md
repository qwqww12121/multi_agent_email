# V5 2026-10-30 diagnostic

Package 0.5.1; frozen Llama-3.1-8B Base; torch 2.10.0+cu126; Transformers 4.57.1; tokenizers 0.22.2; LM Format Enforcer 0.11.3. The diagnostic ended on 2026-10-30 at 15:07:19 Beijing time. Scoring signature: 04088e250ace46a56a6d15235e3d14fd903bc9459f16bd669f1adac7afe846f2. Classification run signature: 5b4ab1fa0edaf433e2d3768f523ad4e8ae0c1574b960984965102e187f428877.

## Sample and execution

Four datasets, 500 mails each: PhishingPot, Enron, ePhishLLM and Kaggle. The exact IDs/order of the earlier frozen 2026-10-29 sample were reused after file-hash checks. Input adaptation changed only protocol/domain identifiers and explicit empty recipient metadata; content was not resampled or relabeled. This is a retrospective previously examined test pool, not a new blind heldout set.

First, two mails per source ran joint classification and four-role evidence analysis. Then all 2,000 mails ran an independent classification-only diagnostic. All 2,000 identities were present, 1,999 mails had all four scores, and one positive PhishingPot mail exceeded the 18,432-token complete-input budget. It remains an unscored record in the overall denominator. No calibration, PPO, RAG, test-driven threshold tuning or answer repair was run.

## Results and metric definitions

At threshold 0.5, uncalibrated equal-mean accuracy over all 2,000 mails was 46.80%. Per-source accuracy: PhishingPot 43.40%, Enron 49.00%, ePhishLLM 55.00%, Kaggle 39.80%. These are diagnostic scores, not trained PPO performance.

Accuracy uses all samples; precision/recall/F1/FNR/negative recall/FPR use scored cases, with unknown support reported separately. Brier and ECE use scored cases. ECE uses 10 equal-width bins on positive-class scores. Source-level ROC-AUC is provided as an additional ranking diagnostic; mixed-task pooled AUC/ECE do not replace source-specific results. Pairwise error correlation is Pearson correlation of binary error indicators, and joint error rate is both-wrong count divided by common scored cases.

The auxiliary test had 32 role reports and 53 attempts including 21 retries. Strict JSON validity was 28/53; initial validity was 19/32. Final valid fields were 66/152, of which 33 had nonempty original quotes. Twelve role reports fully passed structural/quote validation, four partially passed, and 16 were unreliable; 22 lacked valid nonempty quotes. Two of eight mails passed those checks for all four roles. These counts do not certify semantics.

Evidence Accuracy and Role Consistency require complete semantic annotations and are not estimated. PPO gain is unavailable because no PPO model was run. Classification/Brier/AUC and error-complementarity calculations were cross-checked with scikit-learn and NumPy; 209 comparisons matched at absolute tolerance 1e-12. JSON Schema results were independently checked with jsonschema.Draft7Validator.

## Reproduction artifacts

The private diagnostic archive has SHA-256 cab836553fd488ea32be56497caa0d903fca84f889845d5f80015edb1739a4d3; it is not published. The frozen-sample scripts and offline analysis script preserve the original run layout. Users must supply matching original artifacts. Published JSON/HTML/Canvas views contain aggregate metrics only, without raw emails, per-mail IDs, labels or server credentials.
