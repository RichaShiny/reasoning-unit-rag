# Query fusion and original-question retention

## Why these are experimental controls

The matrix BM25 scorer sums nonnegative contributions over unique query terms. If a subquery's terms are a subset of the original question's terms, its score cannot exceed the original question's score for any candidate. Retaining the original and taking raw maximum scores therefore leaves every candidate's score unchanged. A null result in this setting does not establish that decomposition is unhelpful in general.

Dense scores do not have this subset property. Comparing BM25 and dense retrieval under only raw maximum fusion can confound retriever behavior with fusion behavior. This change exposes that choice; it does not designate either fusion policy as universally superior.

## Controls

- `--fusion max` preserves the previous raw-maximum policy and remains the default.
- `--fusion rrf --rrf-k 60` sums `1 / (rrf_k + rank)` across effective queries. Ranks start at one. Exact score ties share competition ranks (1, 1, 3); an all-tied query contributes a constant. Final fused-score ties use the existing stable candidate order. This tie-aware convention is recorded here and in the implementation; it can differ from RRF implementations that arbitrarily order ties.
- `--original-question as-provided` preserves the query list supplied for each non-original condition. `include` adds the original once; `exclude` removes exact stripped-string matches. Neither policy detects semantic paraphrases. The original baseline always contains only the original question.
- `--query-types original llm` selects conditions and requires them on every question. The original baseline is mandatory. The default runs all available conditions. This is useful when the rule generator returns only the original question and cannot support an exclusion experiment.

Empty or malformed query lists, nonfinite scores and inconsistent backend score dimensions fail explicitly. Leading/trailing query whitespace is stripped. Duplicate query strings are collapsed before scoring so repeated annotations do not receive extra RRF weight. If exclusion leaves no query, the run fails before loading models or writing predictions; it does not silently substitute an original-question fallback. Repair the generator/annotations on development data rather than dropping failed questions from a reported evaluation.

## Example ablation

Provide question-only annotations for every input ID, e.g. `{"q": {"llm": ["first subquestion", "second subquestion"]}}`. With the same input, annotations, tokenizer, K and B, run:

```sh
python3 src/diagnostic_matrix.py --dataset hotpotqa \
  --input results/replication-100/examples.json --queries queries.json \
  --query-types original llm --retriever bm25 --fusion rrf \
  --original-question exclude --top-k 5 --context-budget 512 \
  --output results/bm25-rrf-without-original
```

Use distinct output directories for the max/RRF and include/exclude cells. The example uses a **word cap**; pass a fixed reader tokenizer and revision for token budgets. The global top-K-then-pack policy remains unchanged. Actual context usage can differ even with identical caps.

Predictions record effective queries, query count, whether the original is present, fusion settings, retention policy and scored query-candidate pairs. The last count measures scoring work, not token usage or wall-clock cost. `examined_candidates` still refers only to candidates considered by the context packer, not all scored candidates. The manifest records fusion settings, selected query types and the fusion module's source hash.

`scripts/compare_matrix.py` estimates retriever interactions only when effective queries and fusion/retention policies agree. It rejects different settings and legacy cells missing this provenance. Fusion ablations are separate interventions; that script must not mislabel their differences as retriever effects. This change does not yet provide a fusion-ablation inference command or bind externally supplied reader answers to context hashes.

## Validation and interpretation

Regression tests demonstrate BM25 subset-query dominance, RRF scale invariance and tie behavior, exact original retention, malformed-score rejection, CLI lifecycle records and comparison rejection when queries or fusion settings differ. These establish evaluator behavior, not empirical retrieval gains. Existing benchmark artifacts are not overwritten or retroactively given new provenance.

A compatibility replay on the committed 100-question HotpotQA sample verified identical retrieval, scores, metrics and reader inputs in all 600 default BM25 cells against the previous evaluator. Fifteen original-query cells changed only leading/trailing query whitespace. This does not establish dense-encoder equivalence or measure a new fusion gain.
