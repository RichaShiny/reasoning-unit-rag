# Experiment 005: Oracle Passage Retrieval
# Refers to: -oracle_reasoning_unit_retrieval.py
Oracle reasoning units were evaluated on five manually selected questions.

Both baseline and oracle retrieval achieved perfect passage-level recall:

- Baseline recall: 1.000
- Oracle recall: 1.000

Oracle retrieval reduced average retrieved context from 418.8 to 390.5 words.

However, these examples were originally selected because sentence retrieval failed despite successful passage retrieval. Therefore, passage-level recall was already saturated and could not test whether oracle decomposition improves exact evidence retrieval.

The next experiment will evaluate oracle reasoning units at the sentence level.