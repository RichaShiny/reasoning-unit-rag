# Experiment 008: Corrected sentence baselines

Date: October 3, 2026. Status: completed.

## Setup

Evaluate the original-question sentence retriever and existing question-only rule-based queries with the same final budget of five unique sentences. Rule scores use maximum cosine similarity over the original question and generated queries. No prompt/rule changes were made between runs.

- Dataset: HotpotQA distractor validation, revision `1908d6afbbead072334abe2965f91bd2709910ab`.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Evaluator code: commit `cda7bc9cc28b249e88e88041b5fd2bedb3bde7da`.
- Replication: ordered first 100 validation examples (79 bridge / 21 comparison).
- Prospective development check: 500 seeded examples, seed 42, excluding the first 100 (404 bridge / 96 comparison). The two samples have no overlapping IDs.
- Saved examples, IDs, package versions, predictions, diagnostics, and paired analysis are in `results/replication-100/` and `results/validation-500/`.

## Results

| Sample | Method | Exact fact recall | Complete evidence | Page coverage | Mean words |
|---|---|---:|---:|---:|---:|
| First 100 | Original question | 0.6390 | 0.330 | 0.725 | 123.13 |
| First 100 | Rule based | 0.6390 | 0.330 | 0.725 | 123.16 |
| New 500 | Original question | 0.6001 | 0.258 | 0.696 | 121.724 |
| New 500 | Rule based | 0.5994 | 0.256 | 0.696 | 121.720 |

The paired rule-minus-original recall difference on 500 examples is -0.000667, with a question-level percentile bootstrap 95% interval of [-0.002, 0.000] (10,000 resamples, seed 42). Complete-evidence difference is -0.002, interval [-0.006, 0.000]. These intervals describe variation under resampling the observed questions; they do not establish equivalence or future-model performance. Only one example changes recall, so the interval is discrete and cannot establish a broad negative effect.

On the new sample, original-question bridge recall is 0.5634 and comparison recall is 0.7543. Complete-evidence rates are 20.05% and 50.00%, respectively. This identifies bridge questions as the weaker stratum for the next study; it is descriptive, not a controlled comparison of question difficulty.

## Indexing and artifact checks

The first-100 run exactly reproduces Experiment 001's sentence recall (0.639) and rounded word count (123.1). All 20 previously saved failures have the same recall under the corrected evaluator.

There is one blank sentence in the first 100 contexts and three in the new 500. All are trailing: none shifts a nonempty sentence's coordinate under the old filtering code. All gold coordinates point to nonempty source sentences. Thus the indexing fix is necessary for correctness in general, but it does not account for historical recall on these samples.

The audit verifies each retrieved coordinate/text against its saved full context, unique-coordinate and Top-K constraints, gold-coordinate validity, reconstructed evidence metrics, sample/manifest IDs, and summary consistency. These are checks of the completed run artifacts, not a reproduction of the original passage/window experiments.

## What the rules actually do

The generator adds queries for only 5/100 replication questions and 5/500 new questions. It changes two and four final sentence sets respectively. No question's recall improves; one new comparison question loses a supporting fact. The rules' lack of coverage is a measurable limitation, so these results should not be generalized to stronger decomposition methods.

Among the 371 incomplete original-question results on the new sample:

- 284 miss a gold fact on a page absent from the retrieved sentence set.
- 87 miss exact facts despite all missing-fact pages being represented.

This supports measuring both page discovery and within-page selection in the next experiment. It does not prove that iterative retrieval will repair those failures.

## Reproduce

Run from the repository root with the model/dataset dependencies installed. Use new output directories so the published artifacts are preserved:

```sh
python3 src/run_evaluation.py --first --count 100 --top-k 5 --dataset-revision 1908d6afbbead072334abe2965f91bd2709910ab --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --output results/replication-100-rerun
python3 src/run_evaluation.py --count 500 --seed 42 --exclude-first 100 --top-k 5 --dataset-revision 1908d6afbbead072334abe2965f91bd2709910ab --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --output results/validation-500-rerun
python3 scripts/audit_run.py results/replication-100-rerun results/validation-500-rerun
python3 src/paired_analysis.py results/validation-500-rerun/predictions.jsonl --output results/validation-500-rerun/paired_analysis.json
```

The paired-analysis command rejects duplicate/missing example IDs and invalid metric values and records the predictions' SHA-256 hash. It resamples differences from the same question, rather than independently resampling each method. It creates a new output file and refuses to overwrite it.

## Limits and next step

This is sentence retrieval among supplied distractor contexts; it is not full-Wikipedia retrieval or final answer evaluation. Passage and window baselines were not rerun here. Top-K is controlled, but context word budgets were not swept. The first 100 are replication data, and the new 500 are development evaluation data; once inspected here, neither should be described as untouched in later studies. Package versions are captured, but hardware/device and numerical backend were not captured by the current runner, so exact floating-point scores may vary across machines.

Next: implement a strictly question-only automated decomposition adapter, with cached outputs and malformed-output fallback, then evaluate it on a fresh preselected development sample. Use the current two samples for debugging and failure diagnosis rather than claiming a fresh held-out test on them.
