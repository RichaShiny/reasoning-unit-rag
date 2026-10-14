# First real information-control development pilot

See [the complete pilot report](../results/information-pilot/README.md) for selection, annotations, all four planned runs, actual context costs, paired intervals and reproduction instructions.

This pilot uses eight eligible questions from the first 12 bridge-labeled records in the previously inspected 100-question HotpotQA snapshot. All exclusions and ambiguities are recorded. The same Codex agent authored and source-checked the annotations; independent review is pending.

With max fusion and a 512-word cap, passage supporting-sentence recall is 0.7083 for the original question, 0.7292 for question-only decomposition, and 1.0000 for both original-plus-bridge and decomposed-plus-bridge queries. In this small sample the entity-supplied original question matches the decomposed oracle condition. Other units and RRF show different patterns, including losses; the report preserves them all.

These are exploratory retrieval results, not answer accuracy or held-out evidence. No dense model or reader was run. Do not put these small-sample numbers in the paper abstract as a general benchmark finding.
