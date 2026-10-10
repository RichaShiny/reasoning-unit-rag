# Separate sentence evidence from paragraph coverage

MuSiQue supplies supporting-paragraph labels, not the sentence coordinates used by HotpotQA and 2Wiki. Previously the matrix emitted `evidence_recall` and `complete_evidence` for all three datasets. Retrieving an unrelated sentence from every supporting MuSiQue paragraph could therefore produce `complete_evidence: 1` without establishing that the required facts were present.

## Metric schema version 2

| Label level | Primary metric | Completeness metric | Interpretation |
|---|---|---|---|
| Sentence (HotpotQA, 2Wiki) | `evidence_recall` | `complete_evidence` | Fraction/all of the annotated supporting sentence coordinates recovered |
| Paragraph (MuSiQue) | `supporting_paragraph_coverage` | `all_supporting_paragraphs_touched` | Fraction/all of the labeled supporting paragraphs touched by at least one retrieved unit |

MuSiQue outputs no longer contain `evidence_recall` or `complete_evidence`. Paragraph coverage is a proxy for locating source material; it does not certify recovery of facts inside a paragraph. Even a value of one leaves sentence/fact support unverified. Sentence-level annotation or a separately validated fact evaluation is required to make that stronger claim. Conversely, recovery of all annotated HotpotQA/2Wiki sentences is not proof that a reader answers correctly.

`page_coverage` remains available for compatibility and measures coverage of gold document IDs. For paragraph-labeled data it is numerically identical to `supporting_paragraph_coverage`. `context_cost` and optional answer EM/F1 retain their definitions. Retrieval, query fusion and packing are unchanged.

Predictions, summaries and manifests now contain `metric_schema_version: 2`, `evidence_level`, `primary_metric` and `evidence_interpretation`. The manifest also hashes the metric implementation. Query effects still report each available metric. Retrieval-unit interactions use the explicitly named primary metric. Retriever comparisons label each effect with its metric, check the same contract on both sides and check that manifests agree with predictions.

## Aggregation and migration

A summary or retriever comparison must contain one evidence level and one metric schema. Mixed sentence/paragraph records, mislabeled metrics, missing contract fields, inconsistent metric sets, duplicate question IDs and nonfinite values fail explicitly. Keep dataset results separate; do not pool sentence recall with paragraph coverage under a generic retrieval score.

Regenerate legacy diagnostic-matrix predictions and summaries from their original inputs and settings in a new output directory. The analysis functions intentionally reject records lacking the version-2 contract rather than guess their semantics. Existing saved artifacts are not overwritten or relabeled. The older `run_evaluation.py` sentence-only workflow is outside this schema migration.

## Validation

Regression tests cover an irrelevant sentence from a supporting paragraph, partial/full paragraph coverage, exact sentence-coordinate recovery, empty retrieval, mixed/legacy records, paired paragraph effects and a complete MuSiQue CLI-to-comparison round trip. These are evaluator tests, not new benchmark findings.
