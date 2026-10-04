# Experiment 009: Automated decomposition adapter

Status: implementation and offline regression tests completed; live evaluation not run.

## Motivation

Experiment 008 found that the existing rules added queries for only five of 500 new questions. Add a model-based question-only generator before attempting evidence-conditioned retrieval. Query generation and evidence representation remain separate experimental variables.

## Implemented contract

`QuestionGenerator.generate(question)` accepts only a nonempty string. The OpenAI Responses API receives the fixed decomposition instructions and question, an explicit model ID, a token cap, optional reasoning effort, and `store=False`. It receives neither context nor evaluator labels. The SDK is imported only when a live provider is constructed; tests and cache replay need no API credentials.

Accept only a JSON list of one or two nonempty query strings, each no longer than 240 characters. Strip and deduplicate accepted queries. Use the original question plus those queries for max-score fusion, then select the same Top-K/word budget as the original-question and rule-based methods. Invalid text, request errors, and incomplete model responses use the original-question ranking and remain visible as fallbacks.

The request identity determines the cache key; cache entries include raw output and provenance. Cache-only runs require every entry to exist and match. Cached usage/latency represent the original generation; summary call and token totals count only fresh attempts, with missing usage explicitly counted. Failed responses are cached to keep a replay reproducible.

## Validation

Regression tests cover strict parsing, output fallback, exception-message omission, incomplete responses, cache hits, model/config cache separation, tampered prompt identity, question-only inputs under altered gold labels, shared budgets, unchanged baseline/rule results, SDK request parameters, and a complete cache-only CLI run with a synthetic encoder/provider. Those fixtures validate plumbing and do not measure an LLM's retrieval quality.

The previously committed 100- and 500-example baseline artifacts still pass coordinate/text, metrics, and summary reconstruction checks. The `--exclude-ids` option accepts prior manifests or explicit ID lists and records the excluded IDs in the new manifest, enabling a fresh sample outside both inspected cohorts.

## Next live study

First debug generation on inspected examples. Choose and freeze a supported generator snapshot and settings, then create a fresh seeded sample excluding the original 100 prefix and the 500 IDs in `results/validation-500/manifest.json`. Keep the unchanged baselines in the same run. Report automated recall, complete-evidence rate, page coverage, paired intervals, generator fallback rate, subquery coverage, and token usage.

Question-only generation can still introduce memorized or invented entities. Audit that behavior separately instead of claiming the input boundary establishes semantic faithfulness. Generated queries and caches should be saved for review. Only proceed to evidence-conditioned retrieval after this comparison is interpretable.

No OpenAI API key was configured in the implementation environment, so no live request or automated performance result is claimed. The [official text-generation documentation](https://developers.openai.com/api/docs/guides/text) was checked for the Responses API request/output interface.
