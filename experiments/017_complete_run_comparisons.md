# Require completed, comparable diagnostic runs

Two interrupted retrieval runs can have the same saved prefix of question IDs. Comparing just those predictions previously passed the pairing checks and could present a partial sample as an experimental comparison. The comparison CLI now validates each saved run before computing intervals.

## Required artifact contract

`load_matrix_run` reads the prediction file and its adjacent `manifest.json`. It requires:

- Lifecycle status `complete`, an integer processed count equal to the prediction count, and the exact ordered IDs recorded in the manifest.
- Nonempty, unique question records with the versioned metric contract from experiment 015.
- Every passage/sentence/window cell for every manifested query condition, including the original baseline.
- Agreement between cell and manifest fusion, rank constant and original-question retention settings.
- Explicit input/source hashes, metric/fusion source hashes, corpus scope, annotation identity, budget definitions and required configuration fields. Missing fields cannot compare equal merely because both files omit them.

The CLI then checks that both runs use the same input, annotations, dataset, budgets, tokenizer, query conditions, fusion policy, metric contract, corpus scope and recorded evaluator/fusion/metric source hashes. Retriever/model settings may differ: that is the intervention under study. The record-level comparison continues to require identical effective query lists.

Runs made with different evaluator versions must be regenerated under one version before this retriever-only comparison. Even an innocuous source edit changes its hash; this intentionally conservative rule avoids silently treating code changes as retriever effects. The hashes are recorded provenance, not signed attestations or a complete environment lockfile. Matching them does not prove that every dependency and external input was identical.

## Usage and output

```sh
python3 scripts/compare_matrix.py results/left/predictions.jsonl results/right/predictions.jsonl \
  --resamples 10000 --seed 42 --output comparison.json
```

The report records the bootstrap seed/resample count and identifies both prediction paths, example counts, input/evaluator hashes and retriever/model revisions. Effects retain their metric names and paired intervals. An invalid run or mismatch fails before creating the output report. Existing outputs are never overwritten.

The in-memory `compare` function still operates on already supplied records; it cannot certify lifecycle status without manifests. Use the CLI or explicitly call the artifact loader for persisted experiments. This comparison estimates retrieval-primary-metric interactions, not answer-score or fusion-policy effects.

## Validation

Tests cover started/failed/interrupted statuses, truncated or duplicated records, missing cells, manifest-ID/count inconsistencies, absent provenance, cell/manifest disagreement, source-version drift and the complete CLI path with a recorded seed. These checks do not run a new retrieval benchmark or change retrieval scoring.
