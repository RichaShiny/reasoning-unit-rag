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