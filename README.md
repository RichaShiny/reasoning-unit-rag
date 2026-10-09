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

## Local seq2seq decomposition

Use `--generator-provider local` to run a Hugging Face encoder-decoder model without API credentials. Install `requirements-local-generator.txt` if needed. Model loading is lazy at the provider boundary, uses safetensors, disables remote code, and requires an explicit `--generator-revision`. The same strict JSON parser, fallback behavior, and retrieval budgets apply to local and API generation; malformed local output is not silently rewritten into successful queries.

For a small integration pilot on already inspected questions:

```sh
python3 src/run_evaluation.py --first --count 5 --generator-provider local --generator-model google/flan-t5-small --generator-revision 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab --generator-device cpu --generator-max-output-tokens 64 --generator-cache .cache/flan-small-pilot --output results/local-pilot-rerun
```

The [FLAN-T5 model card](https://huggingface.co/google/flan-t5-small) describes the encoder-decoder loading interface. This small model is a diagnostic integration baseline; support for local loading does not imply it produces faithful decomposition. CPU is the default. Greedy decoding uses one beam and no sampling; change `--generator-beams` explicitly for a beam-search ablation. Prompts exceeding `--generator-input-limit` fail into the logged fallback instead of truncating the question. Token-cap output without EOS is marked incomplete.

Local cache identity includes provider, revision, device, input limit, beam count, and decoding settings, keeping local responses separate from API responses. Replay with the same settings plus `--generator-cache-only` does not load model weights. Summaries report fresh provider calls/tokens separately from API calls, so local generation is never presented as an API expense. Use the exclusion and pinned dataset/embedding settings from the automated-evaluation section for a subsequent fresh development study.

## Run startup and failure records

The evaluator now creates its output directory and manifest before importing/loading generator or dataset dependencies. It prints flushed phase messages (`loading_generator`, `loading_dataset`, `loading_embedding_model`, `evaluating`) and progress every ten examples. A slow import therefore leaves a visible phase and a manifest rather than an empty terminal with no run record.

New manifests record Python/platform, hashes of the evaluation source files, the embedding device when available, elapsed time, and the number of predictions saved. Caught startup/evaluation exceptions mark the run `failed`; Ctrl+C marks it `interrupted`, preserving partial predictions and the last phase. Error types are recorded without exception messages. A hard kill or power loss can leave `started`; those runs remain incomplete. Existing output directories are still never overwritten, and manifest updates use atomic file replacement. This reports startup problems; it does not repair slow or incompatible installed dependencies or implement run resumption.

## Completed local pilot

[Experiment 010](experiments/010_local_generator_pilot.md) records an actual five-question FLAN-T5-small run and offline cache replay. All five raw outputs failed the structured-query contract and fell back to original-question retrieval. This validates the failure/replay path; it supplies no successful-decomposition performance result. Raw responses and exact inputs are saved for reviewing the next prompt/model ablation.

## Controlled diagnostic matrix

The measurement-study extension crosses retrieval units and query types under
shared global retrieval and context caps, with BM25, Contriever and configurable
dense encoders. See [the protocol](experiments/011_factorial_diagnostics.md) for
benchmark adapters, answer evaluation, paired confidence intervals and limitations.
Run `python3 src/diagnostic_matrix.py --help` to start.
