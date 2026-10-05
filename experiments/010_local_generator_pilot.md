# Experiment 010: Local generator integration pilot

Date: October 4, 2026. Status: actual local generation and offline cache replay completed.

## Scope and environment

Use the first five already inspected examples from `results/replication-100/examples.json` (three bridge, two comparison). This is an integration/debugging sample, not held-out performance evidence. The saved sample retains lineage to the dataset revision recorded in the original replication manifest.

- Generator: `google/flan-t5-small`, revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`.
- Generator device: CPU; greedy decoding, one beam, no sampling.
- Input limit: 512 tokens; output limit: 64 tokens.
- Embeddings: MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Budget: five unique sentences, no word cap.
- Prompt/parser: unchanged shared JSON-list contract from the automated adapter.
- Runtime/package/source/device details: saved in each manifest.

The virtual-environment import diagnostic was interrupted while `transformers` scanned installed-package metadata. The base Python environment successfully imported the model classes and completed the pilot using the saved dataset sample. This avoids the additional virtual-environment dataset dependencies; it does not establish a permanent fix for the prior stall. The pilot required a model download on the first run; offline replay used the completed caches.

## Observed generator outputs

| Example ID | Raw generated text | Parser result |
|---|---|---|
| `5a8b57f25542995d1e6f1371` | `no` | invalid output |
| `5a8c7595554299585d9e36b6` | `list(map(int, input().split()))` | invalid output |
| `5a85ea095542994775f606a8` | `listase[::-1]` | invalid output |
| `5adbf0a255429947ff17385a` | `list(map(int, input().split()))` | invalid output |
| `5a8e3ea95542995a26add48d` | `list(map(int, input().split()))` | invalid output |

All five generations completed with EOS, but none satisfied the one-or-two-query JSON-list contract. These outputs were retained as text and never executed. All five automated retrieval calls therefore fell back to the original question.

## Retrieval and interpretation

| Method | Exact fact recall | Complete-evidence rate | Mean retrieved words |
|---|---:|---:|---:|
| Original question | 0.6467 | 0.400 | 109 |
| Rule based | 0.6467 | 0.400 | 109 |
| Automated (100% fallback) | 0.6467 | 0.400 | 109 |

The identical automated metrics are a mechanical consequence of fallback, not evidence for or against successful decomposition. No usable-query case exists in this pilot, so a retrieval-effect estimate for usable decomposition is unavailable. The five selected questions also cannot establish a population-level generator failure rate.

Fresh generation made five local provider calls and zero API calls, reporting 753 prompt tokens and 63 generated tokens in total. The summed generation-call latency was about 1.32 seconds on this run, excluding model startup/download and embedding work. Do not treat this one-run timing as a benchmark.

The [FLAN-T5-small model card](https://huggingface.co/google/flan-t5-small) supports its use as a local encoder-decoder baseline. This pilot shows that the particular shared prompt and small checkpoint do not produce usable structured subqueries on these examples. It does not distinguish model capacity from prompt/format sensitivity, and does not generalize to larger models or different prompts.

## Replay and artifact validation

`results/local-pilot-5/` contains the exact examples, full predictions, summaries, source/runtime manifest, diagnostic audit, raw generator cache, and replay verification report. `results/local-pilot-5-replay/` contains the completed cache-only run. All five replay requests hit the exported cache, with zero new provider/API calls.

Verified identical IDs, raw responses, queries, metrics, and retrieved rankings. Score comparisons used an absolute tolerance of 1e-6 on the same machine. Both runs pass the existing source-coordinate/text, deduplication, metric, and summary artifact audit.

## Reproduce with preserved inputs

From the repository root, use the working Python environment and a new output directory:

```sh
HF_HOME="$PWD/.cache/huggingface" python3 -u src/run_evaluation.py --input results/local-pilot-5/examples.json --first --count 5 --top-k 5 --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --generator-provider local --generator-model google/flan-t5-small --generator-revision 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab --generator-cache results/local-pilot-5/generator-cache --generator-cache-only --generator-max-output-tokens 64 --output results/local-pilot-rerun
python3 scripts/audit_run.py results/local-pilot-rerun
```

Replay needs the embedding weights; add `HF_HUB_OFFLINE=1` only after those weights are cached. To regenerate rather than replay, omit `--generator-cache-only` and point `--generator-cache` to a new cache directory. Keep published response caches intact.

## Next experiment

On inspected debugging examples, compare a frozen local few-shot prompt with the current prompt and/or a larger checkpoint. First report valid-query yield and review whether queries express separate information needs rather than answers, code, or invented entities. Do not scale to a fresh retrieval study until the generator produces enough usable queries to make the comparison interpretable. Preserve output-format policy as an explicit versioned variable rather than silently changing how historical responses are parsed.
