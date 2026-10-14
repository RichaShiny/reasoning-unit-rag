# HotpotQA information-control development pilot

**Exploratory results, not paper-ready evidence.** Eight eligible questions from the first 12 bridge-labeled records in the already inspected 100-question replication sample. Annotations were drafted and source-checked by the same Codex agent; independent human review is pending. No question was selected by retrieval success or failure.

## Selection and annotations

The source contains 100 questions: 79 labeled bridge and 21 comparison. The first 12 bridge-labeled records were screened in source order. Four were excluded: two lacked a clear named-intermediate slot under this pilot protocol, one was an explicit age comparison despite its dataset label, and one asked for a country while its gold evidence concerned a county. All 12 candidates and reasons remain in the review files. No replacement questions were added.

Question-only template/description drafts were saved before the source-annotation stage. Intermediate entities were then checked against exact supplied gold sentence coordinates; citations and ambiguity notes are retained. This is a recorded two-stage agent workflow, not a blinded independent annotation study. Some included questions have multiple plausible answer formulations or other ambiguities. Eligibility and annotations must be reviewed before extending this design.

## Fixed design

- BM25 only, using each question’s supplied distractor candidate pool; not open-domain retrieval.
- Four conditions: original O, decomposition D, original plus bridge entities O+E, paired decomposition plus entities D+E.
- Three retrieval units: passage, sentence, centered window. Global K=5; context caps of 128 and 512 **words**, including serialized titles/separators.
- Max fusion and tie-aware RRF with rank constant 60. All four planned runs retained.
- Paired question bootstrap: 2,000 resamples, seed 42. Eight questions per run; 384 total condition records across the four runs, not 384 independent examples.
- No answer reader, LLM-generated decomposition, dense retriever, MuSiQue or 2Wiki result is included.

## Supporting-sentence recall

Each entry is the mean question-level recall of annotated sentence coordinates. Shared caps do not imply equal actual context usage.

| Fusion | Word cap | Unit | O | D | O+E | D+E |
|---|---:|---|---:|---:|---:|---:|
| max | 128 | passage | 0.5833 | 0.5000 | 0.7083 | 0.7500 |
| max | 128 | sentence | 0.7292 | 0.6875 | 0.7500 | 0.7292 |
| max | 128 | window | 0.5208 | 0.4792 | 0.6875 | 0.7917 |
| max | 512 | passage | 0.7083 | 0.7292 | 1.0000 | 1.0000 |
| max | 512 | sentence | 0.7292 | 0.6875 | 0.7917 | 0.7292 |
| max | 512 | window | 0.7083 | 0.6458 | 0.8333 | 0.9167 |
| rrf | 128 | passage | 0.5833 | 0.3958 | 0.7083 | 0.5625 |
| rrf | 128 | sentence | 0.7292 | 0.5000 | 0.7500 | 0.6875 |
| rrf | 128 | window | 0.5208 | 0.3958 | 0.6875 | 0.4375 |
| rrf | 512 | passage | 0.7083 | 0.6250 | 1.0000 | 0.9167 |
| rrf | 512 | sentence | 0.7292 | 0.5000 | 0.7917 | 0.6875 |
| rrf | 512 | window | 0.7083 | 0.4375 | 0.8333 | 0.5833 |

## Paired contrasts at the 512-word cap

Values are mean recall differences with question-level percentile 95% bootstrap intervals. These intervals describe this small selected development sample; they do not justify population claims or correct for the many exploratory contrasts. Degenerate intervals such as [0,0] reflect no observed variation in eight cases, not proof of zero population effect.

| Fusion | Unit | D−O | (D+E)−(O+E) | Interaction |
|---|---|---|---|---|
| max | passage | +0.0208 [-0.1250, +0.1875] | +0.0000 [+0.0000, +0.0000] | -0.0208 [-0.1875, +0.1250] |
| max | sentence | -0.0417 [-0.1250, +0.0000] | -0.0625 [-0.2292, +0.0833] | -0.0208 [-0.2083, +0.1667] |
| max | window | -0.0625 [-0.2917, +0.1458] | +0.0833 [+0.0000, +0.2500] | +0.1458 [-0.1042, +0.4167] |
| rrf | passage | -0.0833 [-0.2708, +0.1250] | -0.0833 [-0.2500, +0.0000] | +0.0000 [-0.2708, +0.2500] |
| rrf | sentence | -0.2292 [-0.3958, -0.0625] | -0.1042 [-0.2500, +0.0000] | +0.1250 [+0.0000, +0.2917] |
| rrf | window | -0.2708 [-0.5208, +0.0005] | -0.2500 [-0.4375, -0.0828] | +0.0208 [-0.1875, +0.2500] |

## What this pilot suggests testing next

With max fusion and the 512-word cap, passage recall was 0.7083 for O, 0.7292 for D, and 1.0000 for both O+E and D+E. In these cases, supplying the bridge entities without decomposing the question already recovered all annotated support. This does not demonstrate that decomposition is useless; it shows why the original-plus-entities control is necessary.

For sentence retrieval at the same cap, recall was 0.7292 for O and 0.6875 for D under max fusion; D fell to 0.5000 under RRF. Both fusion policies preserve the single-query O and O+E rankings, but they combine the two decomposed queries differently. RRF is therefore not an automatic remedy for max-fusion limitations.

One inspectable example is `5a8e3ea95542995a26add48d` (Big Stone Gap). Under sentence retrieval, max-fused D includes the Adriana Trigiani sentence containing the requested location. RRF-fused D instead places more film/Big Stone Gap-related sentences in its five retrieved slots and omits that sentence. In `5a7166395542994082a3e814`, RRF-fused D retains campus-description sentences but loses the Kansas Song sentence. These are observations from saved rankings, not a general explanation established across retrievers.

Actual context cost differs substantially by unit. At max fusion and cap 512, O averages 370.875 words for passages and 125.5 for sentences. The comparison fixes caps, not equal realized context. Query wording, lengths, entity spelling, fusion and packing still need sensitivity analysis.

The next research run should begin with independent annotation review, a larger predefined eligible sample, question-only generated decomposition, and at least one dense retriever. Test whether the observed fusion/bridge pattern persists before revising the abstract. Add a fixed answer reader only through the bound-request workflow; this pilot has no answer-accuracy result.

## Artifacts and reproduction

- `question-stage-review.json`: original drafts and initial eligibility decisions.
- `annotation-review.json`: all candidates, final decisions, exact source citations and ambiguity notes.
- `protocol.json`: selection, frozen input hashes, source commit and planned run settings.
- `examples.json`, `annotations.json`: the exact eight-question input and paired templates.
- `max-b128/`, `max-b512/`, `rrf-b128/`, `rrf-b512/`: manifests, predictions and complete summaries.
- `analysis.json`: machine-reconstructed aggregate results; full effect intervals for all budgets/units.

From the repository root:

```sh
python3 scripts/audit_information_pilot.py
python3 scripts/reproduce_information_pilot.py --output .cache/information-pilot-rerun
```

The auditor verifies input hashes, deterministic sample membership, unchanged question-stage templates, source citations, literal final-answer absence in constructed queries, run completeness and budgets, and reconstructed bootstrap summaries. These checks do not certify semantic fidelity or independent annotation. The reproducer refuses to overwrite an existing output directory.

An actual second execution produced byte-identical predictions, summaries and aggregate analysis for all four runs. Lifecycle timing/path fields in manifests naturally differ. The existing evaluation tests plus pilot input-integrity tests pass (78 tests).

Source data are copied unchanged from the repository’s existing `results/replication-100/examples.json`; its exact SHA-256 and source Git commit are recorded in `protocol.json`.
