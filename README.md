# Reasoning Unit Retrieval for Multi-Hop Question Answering

**Research Work in Progress**

This repository contains my ongoing research investigating whether decomposing complex multi-hop questions into intermediate reasoning units can improve evidence retrieval for Retrieval-Augmented Generation (RAG) systems.

## Motivation

Traditional dense retrieval embeds an entire multi-hop question into a single vector, making it difficult to retrieve all supporting evidence. This project explores whether reasoning-unit decomposition can improve exact evidence retrieval while maintaining retrieval efficiency.

## Research Pipeline

- Baseline passage retrieval
- Sentence-level retrieval
- Sliding-window retrieval
- Failure analysis
- Rule-based reasoning-unit generation
- Oracle reasoning-unit retrieval
- Oracle sentence retrieval

## Dataset

- HotpotQA (Distractor)

## Embedding Model

- sentence-transformers/all-MiniLM-L6-v2

## Preliminary Results

| Experiment | Supporting-Fact Recall |
|------------|-----------------------:|
| Passage Retrieval | 0.793 |
| Sentence Retrieval | 0.639 |
| Oracle Sentence Retrieval* | 0.917 |

\*Oracle sentence retrieval was evaluated on manually selected failure cases to establish an upper-bound for reasoning-unit decomposition.

## Repository Structure

```
src/
    baseline_rag.py
    compare_retrieval.py
    evaluate_retrieval.py
    evaluate_windows.py
    failure_analysis.py
    reasoning_units.py
    reasoning_unit_retrieval.py
    oracle_reasoning_unit_retrieval.py
    oracle_sentence_retrieval.py

experiments/
    001_passage_vs_sentence.md
    002_sentence_windows.md
    003_failure_analysis.md
    004_reasoning_unit_retrieval.md
    005_oracle_passage_retrieval.md
    006_oracle_sentence_retrieval.md
```

## Current Status

This research is actively exploring automated reasoning-unit generation using large language models and evaluating its impact on multi-hop evidence retrieval.

## Future Work

- Evaluate LLM-generated reasoning units
- Compare decomposition strategies across multiple embedding models
- Benchmark on additional multi-hop QA datasets
- Develop a complete reasoning-unit retrieval pipeline for RAG systems
