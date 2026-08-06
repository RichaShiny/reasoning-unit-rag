Experiment 1 (Completed)
Compare passage retrieval vs. sentence retrieval.
Result: Passage recall = 0.793, Sentence recall = 0.639.
Key insight: Sentence retrieval is much more context-efficient but loses supporting evidence.


# Experiment 001: Passage vs. Sentence Retrieval

## Date

August 6, 2026

## Question

Does sentence-level retrieval recover supporting evidence more efficiently than passage-level retrieval on multi-hop questions?

## Setup

- Dataset: HotpotQA distractor validation split
- Examples: 100
- Embedding model: `all-MiniLM-L6-v2`
- Similarity: cosine similarity
- Top-K: 5

## Results

| Retrieval unit | Supporting-fact recall | Average retrieved words |
|---|---:|---:|
| Passage | 0.793 | 418.9 |
| Sentence | 0.639 | 123.1 |

## Initial interpretation

Passage retrieval recovered more of the gold supporting evidence, but it retrieved substantially more text.

Sentence retrieval used about 29% as many words as passage retrieval, but its supporting-fact recall was lower by 0.154.

One possible explanation is that a single sentence is often too narrow for multi-hop questions. Relevant evidence may depend on neighboring sentences, article-level context, or bridge information.

The evaluation is not perfectly symmetric. Passage retrieval receives credit when the correct article title is retrieved, while sentence retrieval must recover the exact gold sentence. This may make the passage metric more forgiving.

## Next

- Measure exact supporting-sentence coverage inside retrieved passages.
- Evaluate Top-K values such as 2, 5, 8, and 10.
- Test small multi-sentence windows as an intermediate retrieval unit.
- Inspect failure cases where passage retrieval succeeds but sentence retrieval fails.

