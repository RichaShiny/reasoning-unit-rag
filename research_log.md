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