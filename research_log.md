# Research Log

## August 4, 2026

Started setting up the project environment.

- Created a virtual environment.
- Installed the required libraries.
- Downloaded the SQuAD dataset first to verify everything was working correctly.
- Switched to HotpotQA because it contains multi-hop reasoning questions, which are more suitable for this project.

Implemented two retrieval baselines:

1. Document retrieval
2. Sentence retrieval

Used the first HotpotQA example to compare both approaches.

Question:

> Were Scott Derrickson and Ed Wood of the same nationality?

The document retriever successfully found the relevant Wikipedia articles, but each result contained a large amount of unrelated information.

The sentence retriever returned:

- Scott Derrickson is an American director.
- Edward Davis Wood Jr. was an American filmmaker.

Both approaches retrieved enough information to answer the question correctly. However, the sentence retriever produced a much cleaner context by eliminating most of the unnecessary text.

This is only a single example, so it's far too early to draw conclusions, but it suggests that retrieving smaller reasoning units could reduce context noise without losing the required evidence.

### Notes

One thing I noticed while building this is that HotpotQA already provides the gold supporting facts. That will make it much easier to automatically evaluate whether a retrieval method actually finds the evidence instead of only checking if the final answer is correct.

### Next

- Evaluate both retrieval methods on a larger set of HotpotQA questions.
- Measure how often each method retrieves the gold supporting facts.
- Experiment with retrieval units beyond individual sentences.

## August 6, 2026
## Experiment 001
Ran the first 100-question retrieval evaluation on HotpotQA.

Passage retrieval had higher supporting-fact recall:

- Passage recall: 0.793
- Sentence recall: 0.639

Sentence retrieval was much more compact:

- Passage context: 418.9 words
- Sentence context: 123.1 words

This was not the result I initially expected. Sentence retrieval reduced context substantially, but individual sentences may be too narrow to preserve the evidence needed for multi-hop reasoning.

The next experiment should test sentence windows rather than jumping directly to a new reasoning-unit design.

## Experiment 002

Tested three-sentence retrieval windows.

Results:

- Three-sentence window recall: 0.642
- Average context: 257.6 words

This was almost identical to single-sentence recall, despite retrieving more than twice as much text. Fixed local windows did not meaningfully improve evidence recovery.

The result suggests the issue may not simply be insufficient neighboring context. The retrieval unit may need to represent relationships between evidence rather than contiguous text.


# After Oracle_sentence_retirval:
1. Passage retrieval has good coverage but uses much more context.
2. Sentence retrieval is compact but misses exact evidence.
3. Fixed sentence windows add context without meaningfully improving recall.
4. Weak rule-based decomposition produces no improvement.
5. High-quality oracle reasoning units substantially improve exact sentence recall.
## October 3, 2026

Audited the 20 saved sentence failures: 12 bridge, 8 comparison; mean saved recall 0.5325; 23 missing facts; 10 cases miss evidence from a page absent in retrieved sentences. These selected historical records are internally consistent but source coordinates remain unverified until the indexing fix is rerun.

Added a reproducible standard-library artifact audit and Experiment 007, which connects HotpotQA, Decomposed Prompting, and IRCoT to a prospective evaluation with isolated generator inputs, held-out IDs, paired metrics, and context-budget controls. No new embedding or automated-generation result is claimed.

## October 3, 2026 — corrected replication and development check

Completed pinned dataset/model evaluations with the merged evaluator. The first 100 reproduce sentence recall 0.6390, and the saved 20 failure recalls all match. No nonempty source coordinate is shifted by blanks in either evaluated sample, so the indexing bug did not explain these results.

On 500 seeded validation questions excluding the original prefix, baseline recall is 0.6001 and complete-evidence rate 25.8%. Rule queries are generated for only five questions, with zero recall improvements and one loss; paired recall delta -0.000667, bootstrap 95% interval [-0.002, 0.000]. Among 371 incomplete baseline results, 284 lack a required page and 87 miss exact evidence on represented pages.

Added question-level paired-bootstrap post-processing, artifact audits, actual saved run outputs, and Experiment 008. These 500 examples are now inspected development data; reserve a fresh frozen sample for future automated-generator claims.

## Automated decomposition implementation

Added a strictly question-only generator adapter and evaluator integration with raw-response caching, configuration identity, malformed/incomplete/error fallback, and cache-only replay. The generator model is an explicit setting. Summaries record automated evidence metrics and fresh-call usage/fallback statistics. Added explicit prior-ID exclusions for the next fresh development sample.

Offline tests cover the request boundary, parser, fallbacks, cache identities, shared budgets, unchanged baselines, SDK contract, and full cache-replay CLI flow. Existing saved baseline runs still pass artifact audits. No API key is configured here, so live-generation quality and retrieval gains remain unmeasured. Experiment 009 documents the implementation and next study.
