# Experiment 006: Oracle Reasoning Units for Sentence Retrieval

## Date

August 6, 2026

## Research Question

Can high-quality reasoning-unit queries improve exact supporting-sentence retrieval on multi-hop questions where standard sentence retrieval fails?

## Hypothesis

Manually constructed reasoning units will retrieve more gold supporting sentences than querying with the original question alone.

## Setup

- Dataset: HotpotQA distractor validation split
- Selected examples: 5
- Selection criterion: sentence-retrieval failure cases
- Embedding model: `all-MiniLM-L6-v2`
- Retrieval unit: individual sentence
- Retrieval budget: Top 5 sentences
- Reasoning-unit generator: manually constructed oracle queries

## Results

| Method | Supporting-sentence recall | Average retrieved words |
|---|---:|---:|
| Original-question retrieval | 0.542 | 99.5 |
| Oracle reasoning-unit retrieval | 0.917 | 110.5 |

Recall difference: **+0.375**

## Interpretation

Oracle reasoning units substantially improved exact supporting-sentence recall while increasing retrieved context only slightly.

This suggests that sentence retrieval itself is not necessarily the primary limitation. The more important bottleneck may be the query representation used to retrieve evidence. The original question often compresses multiple unresolved information needs into one embedding, while reasoning-unit queries represent those needs separately.

## Limitations

This was a small proof-of-concept experiment using five manually selected failure cases. The reasoning units were manually written with knowledge of the intended reasoning path, so the result should not be interpreted as performance from an automated system.

## Conclusion

The experiment provides preliminary evidence that high-quality question decomposition can materially improve exact evidence retrieval for multi-hop questions.

## Next Steps

- Expand the oracle evaluation to 20 failure cases.
- Categorize units by bridge, comparison, entity attribute, and relation.
- Evaluate an automated reasoning-unit generator.
- Compare automated units with oracle units and the original question.