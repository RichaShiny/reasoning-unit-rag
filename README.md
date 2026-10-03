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
