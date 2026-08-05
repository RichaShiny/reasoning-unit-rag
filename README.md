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