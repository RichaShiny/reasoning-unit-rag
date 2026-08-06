# Experiment 002: Three-Sentence Windows

## Date

August 6, 2026

## Research question

Does adding neighboring context to individual sentences improve supporting-fact retrieval while remaining more compact than full passages?

## Hypothesis

A three-sentence window will recover more supporting evidence than individual-sentence retrieval because neighboring sentences may preserve local context.

## Setup

- Dataset: HotpotQA distractor validation split
- Examples: 100
- Embedding model: `all-MiniLM-L6-v2`
- Similarity: cosine similarity
- Top-K: 5
- Window size: 3 sentences

## Results

| Retrieval unit | Supporting-fact recall | Average retrieved words |
|---|---:|---:|
| Single sentence | 0.639 | 123.1 |
| Three-sentence window | 0.642 | 257.6 |
| Full passage | 0.793 | 418.9 |

## Interpretation

The three-sentence window produced almost no recall improvement over individual sentences. Recall increased from 0.639 to 0.642, while the average retrieved context more than doubled.

The window remained smaller than full-passage retrieval, but it did not close the recall gap.

This suggests that adding adjacent text is not enough. The missing evidence may not be located in neighboring sentences, or semantic similarity may still fail to identify the reasoning-critical evidence.

Overlapping windows also introduced substantial duplicated context.

## Next

- Test five-sentence windows for completeness.
- Measure duplicate-token overlap across retrieved windows.
- Inspect examples where passage retrieval succeeds but window retrieval fails.
- Design retrieval units based on evidence relationships rather than fixed text boundaries.