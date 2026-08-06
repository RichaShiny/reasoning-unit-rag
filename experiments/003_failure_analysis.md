# Experiment 003: Sentence Retrieval Failure Analysis

## Date

August 6, 2026

## Goal

Understand why passage retrieval recovers supporting evidence more successfully than sentence retrieval on HotpotQA.

## Setup

- Dataset: HotpotQA distractor validation split
- Examples: 100
- Embedding model: `all-MiniLM-L6-v2`
- Top-K: 5
- Failure definition:
  - passage retrieval recovers all supporting article titles
  - sentence retrieval misses at least one gold supporting sentence

## Result

27 out of 100 examples matched the failure condition.

## Initial failure patterns

### Bridge retrieval without answer retrieval

The sentence retriever often identified an intermediate fact but failed to use that fact to retrieve the next required piece of evidence.

Example:

- Retrieved: Lewiston Maineiacs played at Androscoggin Bank Colisée
- Missed: the arena seats 3,677 people

### Unbalanced comparison evidence

For comparison questions, the retriever frequently recovered evidence for one entity while missing evidence for the other.

### Lexical distraction

Sentences sharing names or surface terms with the question sometimes outranked the correct answer-bearing sentence.

### Multi-step evidence chains

Some questions required evidence from multiple articles and multiple supporting sentences. A single retrieval step did not explicitly model the dependency between those facts.

## Interpretation

The results suggest that the main limitation is not simply retrieval-unit size. The system may need to retrieve iteratively based on the current reasoning state.

A possible direction is:

Question → retrieve first evidence → generate unresolved sub-question → retrieve next evidence

This may be more appropriate for multi-hop reasoning than retrieving a fixed Top-K set once.