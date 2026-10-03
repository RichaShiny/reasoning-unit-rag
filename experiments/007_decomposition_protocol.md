# Experiment 007: Evidence coverage and automated decomposition

Date: October 3, 2026. Status: artifact audit completed; prospective retrieval experiment not run.

## What the existing artifacts establish

Run `python3 src/audit_failures.py --output results/failure_audit.json` from the repository root. The audit reconstructs recall and missing facts from saved gold/retrieved coordinates and rejects inconsistent records.

The 20 saved failures contain 12 bridge and 8 comparison questions. Their mean sentence recall is 0.5325, with 23 missing supporting facts. In 10 cases, at least one missing fact belongs to a page absent from the retrieved sentence set. In the other 10, all missing-fact pages are represented, but the required sentences are absent.

These are the first 20 of 27 selected failures from a 100-example run, not a random sample. The audit verifies internal consistency only: source sentence IDs may have shifted when blank sentences were filtered. Rerun after the evidence-indexing fix before treating these categories or earlier recall numbers as definitive. The saved artifact lacks full contexts, so it cannot establish the size of that bug's effect.

Working hypothesis: page discovery and within-page sentence selection are separate failure modes. Query decomposition should be evaluated against both, rather than judged only on mean recall. This is an inference from the selected artifact, not a population-level finding.

## Related work and implication

- [HotpotQA (Yang et al., 2018)](https://aclanthology.org/D18-1259/) supplies sentence-level supporting facts and comparison questions. Preserve source coordinates and distinguish exact evidence coverage from answer correctness. The project's distractor setup ranks supplied candidate pages; it is not full-Wikipedia retrieval.
- [Decomposed Prompting (Khot et al., 2023)](https://arxiv.org/abs/2210.02406) separates complex tasks into modular subtasks, including retrieval in open-domain QA. It motivates an automated decomposition baseline; it does not validate this repository's max-score fusion or sentence units.
- [IRCoT (Trivedi et al., 2023)](https://aclanthology.org/2023.acl-long.557/) interleaves retrieval and reasoning so later retrieval depends on earlier evidence. This motivates an evidence-conditioned second query for bridge questions. Our proposed two-stage ablation below is inspired by that mechanism and is not an IRCoT reproduction.

The project's existing five oracle cases were selected failures, and their hand-written queries include bridge entities. The 0.375 recall improvement is a diagnostic signal, not evidence that question-only automation achieves that gain. Also, query decomposition changes retrieval queries; it does not by itself create a new evidence-unit representation. Keep those two research questions separate.

## Prospective protocol

1. Rerun the original first-100 baselines after fixing IDs, saving every example's predictions and dataset/model revisions. This is replication, not held-out testing.
2. Tune on 100 seeded training examples. Freeze prompts, fusion rule, model revision, and budgets before touching the evaluation set.
3. Exclude the original first 100 validation examples. Select 500 remaining validation IDs with a fixed seed (42), persist those IDs, and run all methods on exactly that set. Report bridge/comparison counts. This is a prospective development evaluation; do not claim official test performance.
4. Compare original-question sentence retrieval, existing rule-based queries, question-only automated decomposition, and an evidence-conditioned two-stage variant. Keep oracle queries in a separate diagnostic table.
5. Automated generation receives only the question. For the two-stage variant, retrieve two sentences using the original question, then generate one unresolved query from the question and those two sentences. Gold facts, gold titles, answers, and failure labels remain evaluator-only. In distractor ranking, candidate titles can occur in indexed text; they must not be passed wholesale to the query generator.
6. Question-only decomposition returns up to two concrete subqueries as a JSON string list. Reject empty or malformed outputs, log the failure, and fall back to the original question. Record the exact generator model, prompt, decoding configuration, response, latency, and token use; never silently drop examples.
7. For single-stage fusion, score each sentence with the maximum cosine similarity over generated queries and the original question; select five unique source coordinates. For two-stage retrieval, retain the initial two and add the top three unseen coordinates from the second query. Use deterministic corpus-order tie breaking. Log total retrieval calls and candidates scored; equal final Top-K does not imply equal compute.

## Metrics and decisions

Primary: paired difference in mean exact supporting-fact recall, with a seeded 10,000-resample question-level bootstrap 95% interval. Also report complete-evidence rate (all gold facts recovered), supporting-page coverage, and bridge/comparison strata. Compute intervals by resampling the same example IDs for both methods.

Report unique retrieved words, final unique sentence count, query-generation cost, and failure/fallback rate. Add a secondary word-budget sweep at 100, 200, and 400 whitespace-separated words: greedily accept ranked unseen sentences that fit, never split a sentence, include title words, and credit only included coordinates. Same Top-K across passages and sentences is not a matched text budget.

A positive recall difference on held-out development examples, with an interval excluding zero and no decrease in complete-evidence rate, would support scaling the decomposition study. Otherwise, inspect page-missing versus within-page misses before changing the evidence representation. These are prospective decision rules, not observed outcomes. Final answer EM/F1 requires a separately fixed answer generator and has not been measured here.

## Next implementation deliverables

- An import-safe evaluator with CLI sampling/budgets and per-example JSONL outputs.
- A generator adapter with strict input isolation and cached responses.
- Regression tests for shared-budget deduplication, empty output fallback, and evidence-conditioned input isolation.
- Corrected replication results, followed by the frozen prospective comparison.
