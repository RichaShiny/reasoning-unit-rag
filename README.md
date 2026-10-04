# Reasoning Unit RAG

## Research Question

Can retrieving reasoning units instead of document chunks improve multi-hop reasoning in Retrieval-Augmented Generation (RAG)?

## Motivation

Most Retrieval-Augmented Generation systems retrieve document chunks. This project investigates whether reasoning units (sentences, claims, or other structured evidence) provide better retrieval quality for multi-hop reasoning tasks.

## Dataset

- HotpotQA

## Current Progress

- [x] Environment setup
- [x] Baseline document retrieval
- [x] Sentence retrieval
- [ ] Evaluate 100 examples
- [ ] Design reasoning units
- [ ] Compare against existing RAG methods

## Repository Structure

src/
Research code

experiments/
Experiment notes

results/
Experimental outputs

papers/
Drafts and figures

research_log.md
Daily research notes


# 06 August 14:00 

So far:

The roles are:
baseline_rag.py → loads and tests the dataset
compare_retrieval.py → compares passage vs sentence retrieval on one example
evaluate_retrieval.py → evaluates passage vs sentence retrieval on 100 examples
evaluate_windows.py → evaluates 3-sentence windows
failure_analysis.py → saves and prints sentence-retrieval failure cases
reasoning_units.py → generates rule-based sub-queries
reasoning_unit_retrieval.py → evaluates rule-based reasoning units
oracle_reasoning_unit_retrieval.py → oracle test at passage level
oracle_sentence_retrieval.py → oracle test at sentence level

## Continuing the research

The next study is specified in [Experiment 007](experiments/007_decomposition_protocol.md), including related work, leakage controls, and evaluation budgets. Audit the existing saved failures without downloading models:

```sh
python3 src/audit_failures.py --output results/failure_audit.json
```

Historical retrieval scores need replication after correcting source sentence IDs; the audit checks saved-record consistency, not dataset-coordinate correctness.

## Reproducible sentence evaluation

`src/run_evaluation.py` compares original-question retrieval with the existing rule-based query generator. It loads datasets/models only when executed; tests use a small deterministic encoder and require only NumPy.

Replicate the first 100 validation examples with corrected source IDs:

```sh
python3 src/run_evaluation.py --first --count 100 --top-k 5 --output results/replication-100
```

Use a seeded sample excluding those examples:

```sh
python3 src/run_evaluation.py --count 500 --seed 42 --exclude-first 100 --top-k 5 --output results/validation-500
```

Add `--word-budget 200` to enforce a title-inclusive word cap as well as Top-K. Whole sentences that do not fit are skipped; overlapping source coordinates are counted once. Score ties use corpus order. Pass `--dataset-revision` and `--model-revision` with commit hashes for pinned runs; unpinned runs are explicitly marked in the manifest. A local JSON list in the same HotpotQA schema can be supplied with `--input`.

Each new output directory contains `manifest.json`, the exact `examples.json` sample, `predictions.jsonl`, and `summary.json`. Existing directories are never overwritten. A manifest marked `started` indicates an incomplete run. Summary metrics include exact sentence recall, complete-evidence rate, page coverage, retrieved words/sentences, question-type strata, and the paired mean recall difference. This runner does not yet implement LLM query generation, bootstrap intervals, or answer accuracy. Saved examples include gold labels for evaluation; only question text enters the rule-based generator.

Run regression checks without dataset/model downloads:

```sh
python3 -m unittest discover -s tests -v
```

## Completed baseline replication

[Experiment 008](experiments/008_corrected_baselines.md) records real pinned-model runs on the first 100 validation questions and 500 further seeded questions. Original-question sentence recall is 0.6390 and 0.6001, respectively; the existing rule-based generator yields no improvement. Full evidence is recovered on only 25.8% of the new 500 questions.

[Run artifacts](results/README.md) include exact samples, predictions, diagnostics, and paired-bootstrap intervals. `src/paired_analysis.py` now supplies intervals as a separate post-processing command; the evaluation runner's summary alone still reports means.

## Automated question-only decomposition

The evaluator can now compare an `automated` method alongside the two existing sentence baselines. The adapter accepts only question text and validates a JSON list of one or two subqueries. All methods share the same final unique-sentence and optional word budget; automated scores use the maximum cosine similarity across the original question and generated queries.

Live generation uses the [OpenAI Responses API](https://developers.openai.com/api/docs/guides/text). Install the optional SDK with `python3 -m pip install -r requirements-generator.txt` and configure `OPENAI_API_KEY` in the environment. The API key is never written to manifests or cache entries. An explicit generator model ID is required; use a snapshot ID when available, and choose an output-token limit and reasoning setting supported by that model. The returned model ID is logged. No live API result is included in this implementation PR.

For a fresh development sample, exclude both the original prefix and the 500 IDs already inspected. Set `GENERATOR_MODEL` to the chosen model ID before running:

```sh
python3 src/run_evaluation.py --count 500 --seed 42 --exclude-first 100 --exclude-ids results/validation-500/manifest.json --dataset-revision 1908d6afbbead072334abe2965f91bd2709910ab --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --generator-model "$GENERATOR_MODEL" --generator-cache .cache/decomposition --generator-max-output-tokens 512 --output results/automated-development-500
python3 src/paired_analysis.py results/automated-development-500/predictions.jsonl --candidate automated --output results/automated-development-500/paired_analysis.json
```

This makes live API requests for cache misses. Freeze the model, prompt, token limit, reasoning setting, sample IDs, and retrieval budgets before interpreting the comparison. Debug on previously inspected examples first. The adapter has no candidate contexts, gold titles/facts, answers, retrieval tools, or prior conversation. Question-only inputs do not prevent a model from using memorized knowledge; assess generated queries for invented bridge entities separately.

Each cache entry records the exact request, raw output, returned model, response status/ID, usage when available, and original latency. Its key includes the question, model, full prompt, output-token cap, reasoning setting, and parser version. Valid cached responses replay without API calls. Malformed output, incomplete responses, and request errors are cached and fall back to the original question with an explicit reason. Exception messages are omitted from logs. There are no SDK retries; use a new cache directory for a deliberate retry experiment. Use a single writer per cache directory.

Add `--generator-cache-only` with the same model/configuration to replay without an API key or the SDK. Missing or mismatched cache entries stop the run rather than silently pretending an automated result exists. Preserve and share the cache alongside evaluation artifacts when reproducing a study; `.cache/` stays out of Git by default.

The run manifest includes the generator prompt/configuration and installed SDK version. Predictions retain fallback and cache status. The summary reports automated retrieval metrics, paired mean recall difference, fallback rate, cache hits, fresh call attempts, latency, and reported token usage for fresh calls only. Failed calls with unavailable usage are counted separately; token totals are not a complete billing estimate. No text-budget sweep or evidence-conditioned second hop is implemented here.
