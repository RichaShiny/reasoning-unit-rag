# Experiment 004: Rule-Based Reasoning Unit Retrieval

## Date

August 6, 2026

## Research Question

Can simple rule-based reasoning unit decomposition improve passage retrieval for multi-hop question answering?

## Hypothesis

Generating intermediate reasoning units before retrieval will retrieve more relevant evidence than using the original question alone.

## Setup

- Dataset: HotpotQA distractor validation split
- Examples: 100
- Embedding model: all-MiniLM-L6-v2
- Retrieval: Top-5 passages
- Reasoning-unit generator: Rule-based heuristics

## Results

| Method | Supporting Fact Recall | Average Retrieved Words |
|---------|-----------------------:|------------------------:|
| Baseline Passage Retrieval | 0.793 | 418.9 |
| Rule-Based Reasoning Unit Retrieval | 0.793 | 418.5 |

Average reasoning units generated per question: **1.09**

Recall difference: **+0.000**

## Interpretation

The rule-based reasoning unit generator produced almost no meaningful decomposition for most questions. As a result, retrieval behavior was nearly identical to the baseline passage retriever.

This experiment suggests that the effectiveness of reasoning-unit retrieval depends primarily on the quality of the reasoning-unit generator rather than the retrieval mechanism itself.

## Conclusion

The retrieval architecture functions correctly, but simple handcrafted rules are insufficient for decomposing complex multi-hop questions. Future experiments should evaluate stronger reasoning-unit generators, such as LLM-based decomposition or oracle reasoning units.

## Next Steps

- Evaluate manually constructed (oracle) reasoning units on failure cases.
- Compare oracle reasoning units against rule-based generation.
- Replace the rule-based generator with an LLM.
- Measure retrieval recall and answer accuracy.