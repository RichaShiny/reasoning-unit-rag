# Factorial measurement protocol

The paper is a diagnostic measurement study: when does query representation help,
and how does that depend on retrieval granularity and retriever?

## Conditions

Run the same benchmark export with BM25, Contriever and a pinned modern embedder.
Within each run cross passages, sentences and centered 3-sentence windows with the
original question, rule decomposition, externally supplied oracle queries, and
externally supplied question-only LLM decomposition. The matrix runner evaluates
all answerable input questions; it never selects failures. Existing failure pilots
remain exploratory. Use a separate held-out split after fixing hyperparameters.

`src/diagnostic_matrix.py` accepts native HotpotQA JSON/Hugging Face exports,
2WikiMultihopQA JSON, and MuSiQue JSONL. These are per-question provided-candidate
experiments, **not open-domain retrieval**. MuSiQue unanswerable examples are excluded
and counted. MuSiQue evidence is paragraph-level: sentence retrieval touching a
supporting paragraph earns `supporting_paragraph_coverage`, not verified sentence support.
The [version-2 metric contract](015_evidence_metric_semantics.md) keeps that proxy separate
from sentence evidence and requires regeneration of legacy matrix outputs for analysis.
Its regex sentence splitter is deterministic but should be audited before final runs.
MuSiQue oracle queries substitute prior gold answers into dependency placeholders;
this is explicitly privileged oracle information, never a deployable query generator.
HotpotQA/2Wiki oracle queries require independent annotations covering all questions.

## Matched budgets and estimands

By default, each candidate unit u and query set Q use s(u,Q)=max over q in Q of s(u,q).
Use `--fusion rrf` to test rank fusion separately; see [fusion controls](014_query_fusion_controls.md).
Rank once and examine only the global top K. Pack intact units in that order under
one shared context cap B, including titles and context separators. No oversized-unit
replacement search and no per-query K multiplication. More queries still cost more
encoding/scoring: matched output budgets do not equal matched compute. Record query
counts and candidate counts. Overlapping windows retain their duplicate context cost.
A shared budget is a cap, not an assurance that every condition consumes equal tokens.
Report actual cost and run K/B sweeps; passage rejection under small B can itself
cause losses. The reader prompt and question tokens are outside B and must be fixed.

Without `--tokenizer` the cap is explicitly **words**, not tokens. For paper runs pass
the same fixed reader tokenizer and immutable revision to every condition. Contriever
uses attention-masked mean pooling, L2 normalization, and a 512-token input limit.
Dense sentence-transformer models use encode_query/encode_document for asymmetric
prompts. Pin revisions and audit truncation for the chosen models.

For sentence-labeled benchmarks measure gold-sentence recall, complete evidence and page coverage.
For MuSiQue measure supporting-paragraph coverage and whether all supporting paragraphs were touched.
Average questions, not facts, and keep these different evidence levels separate. Report question-level percentile bootstrap intervals.
Query effect is the paired query-minus-original difference at a fixed unit/retriever,
using sentence recall or paragraph coverage as explicitly recorded in `primary_metric`.
The unit interaction is (query-original at unit) minus (query-original at passage).
`scripts/compare_matrix.py` computes the corresponding paired retriever interaction
and rejects mismatched questions, effective query lists, fusion/retention policies, annotations or budgets.
Legacy predictions without the new fusion provenance must be regenerated for this comparison. Intervals are exploratory,
uncorrected for multiple comparisons; predeclare primary contrasts before final runs.

## Running

From the repository root, no model dependencies are needed for BM25:

```sh
python3 src/diagnostic_matrix.py --dataset hotpotqa \
  --input results/replication-100/examples.json --retriever bm25 \
  --top-k 5 --context-budget 512 --output results/my-bm25-run
```

Use `--retriever contriever --model facebook/contriever --model-revision COMMIT`
or `--retriever dense --model MODEL_ID --model-revision COMMIT` for dense runs.
Use `--tokenizer READER_TOKENIZER --tokenizer-revision COMMIT` for token budgets.
Supply `--queries queries.json` with this format for every input ID:

```json
{"question-id": {"oracle": ["gold-informed subquery"], "llm": ["question-only subquery"]}}
```

Generate LLM annotations once with the existing question-only `QuestionGenerator`,
freeze them across retrievers and units, and preserve its model/prompt/cache provenance.
Missing condition coverage fails instead of silently selecting a favorable subset.
For final answers, export each cell's `reader_input`, run the same fixed reader,
then provide `--answers answers.json` (ID -> {"passage/original": "answer", ...}) in a
new run. The runner does not call a reader API or invent predictions. EM/token F1
are generic normalized answer metrics; also use native dataset evaluation scripts
before claiming official benchmark scores. Never let the reader see oracle queries
or labels: only the original question and exported retrieved context.

Every run records IDs, configuration, input/annotation/answer hashes, source hash,
per-example results and lifecycle status. Do not overwrite previous runs. The initial
BM25 100-example run is an integration check on the old replication sample, not a
held-out finding. Dense model downloads, full benchmark collection, oracle annotation,
and reader inference remain necessary before paper-level conclusions.

Sources: [MuSiQue](https://github.com/StonyBrookNLP/musique),
[2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop),
[Contriever](https://huggingface.co/facebook/contriever),
[Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/model.html).
