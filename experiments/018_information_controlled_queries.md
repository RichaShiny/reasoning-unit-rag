# Decomposition versus supplied bridge information

The earlier oracle condition combined query restructuring with access to intermediate entities that may be absent from the original question. This protocol constructs a four-condition experiment from shared annotations so their effects can be estimated separately.

| Condition key | Queries | Supplied intermediate entities |
|---|---|---|
| `original` (O) | Original question | No |
| `decomposition` (D) | Decomposition templates rendered with question-only descriptions | No |
| `original_bridge` (O+E) | Original question plus a fixed description/entity list | Yes |
| `decomposition_bridge` (D+E) | The same templates rendered with intermediate entities | Yes |

D and D+E have the same number and order of query templates. Duplicate rendered steps are rejected rather than silently changing their query counts. An unresolved descriptive subquestion can naturally equal the original question; the protocol does not append the original as an extra query. O+E remains one query, with a deterministic `Bridge information:` appendix sorted by slot name.

## Annotation contract

`--information-controls annotations.json` expects:

```json
{
  "schema_version": 1,
  "examples": {
    "synthetic-bridge-1": {
      "question": "Where was the author of The Lantern born?",
      "steps": ["Who wrote The Lantern?", "Where was {author} born?"],
      "bridges": {
        "author": {"description": "the author of The Lantern", "entity": "Mira Vale"}
      },
      "provenance": {"author": "annotator identifier", "source": "annotation source/version"}
    }
  }
}
```

Every evaluated answerable question must have exactly one annotation, with its exact original question text. Extra or missing IDs fail; the runner never quietly selects only annotated successes. Each annotation needs at least two template steps and one bridge slot, with all slots used. Only plain declared placeholders are permitted. The final answer is not an annotation field and is not read by the condition builder.

This structural contract does **not** certify semantic validity. Audit descriptions and template wording using the question alone; obtain intermediate entities independently from a documented gold source; never insert the final answer into a bridge slot or template. Verify that the two renderings express the same subquestions and that descriptive queries contain no privileged entity facts. The code cannot reliably detect leakage, memorized answers, unfaithful paraphrases or mislabeled gold intermediates.

Apply the bridge experiment to a predefined eligible stratum. Comparison questions without a hidden intermediate entity do not fit this protocol: analyze them separately rather than invent bridge labels. Report eligible counts and exclusions against the full benchmark. MuSiQue's existing native oracle queries are not used as a substitute for these paired annotations; the same explicit annotation contract is required for all datasets.

## Running a synthetic smoke test

The committed demo is invented data, not a benchmark result:

```sh
python3 src/diagnostic_matrix.py --dataset hotpotqa \
  --input examples/information-controls/demo-hotpotqa.json \
  --information-controls examples/information-controls/demo-annotations.json \
  --retriever bm25 --fusion rrf --top-k 5 --context-budget 100 \
  --resamples 100 --output .cache/information-controls-demo
```

For real experiments replace both files with frozen benchmark inputs and reviewed annotations, pin retriever/tokenizer revisions, and sweep predeclared budgets and fusion choices. This flag always runs all four conditions at all three retrieval units. It rejects `--queries`, `--query-types`, and non-default original-retention policies to avoid changing the intervention unintentionally. The standard rule/oracle/LLM path remains available without this flag.

Reader request export and bound answer evaluation work on these cells without exposing the oracle query text to the reader. The reader receives only the original question and retrieved context. MuSiQue still reports paragraph coverage, not verified sentence-support recall.

## Paired estimates

`summary.json` adds `information_effects.units[unit][metric]`:

- `decomposition_without_bridge`: D − O.
- `decomposition_with_bridge`: (D+E) − (O+E).
- `bridge_without_decomposition`: (O+E) − O.
- `bridge_with_decomposition`: (D+E) − D.
- `interaction`: [(D+E) − (O+E)] − [D − O].

Each contrast is formed per question before question-level bootstrap resampling. Reports include effects and percentile 95% intervals for the available evidence metrics, context cost and, when genuinely supplied through the bound-reader workflow, answer metrics. Effects on context cost indicate resource changes, not answer improvements. Preserve primary/secondary hypotheses and account for exploratory multiple comparisons.

The manifest records protocol `paired-bridge-substitution-v1`, the complete annotation-file hash and the construction/analysis module hash. Retriever comparisons require those hashes and effective queries to agree. Query counts, total query word counts and query-candidate score counts are logged per cell.

## Interpretation limits

The experiment controls the template construction and where bridge entities enter. It does not prove a uniquely causal decomposition effect: query length, wording, number of retrieval calls and fusion behavior can still differ. Shared K and context caps are not equal compute or equal realized context. D versus D+E is the tightly paired substitution; O+E's appendix is one declared operationalization that needs wording sensitivity analysis. Gold entities measure privileged headroom, not a deployable system.

A positive interaction means decomposition benefits more when bridge information is supplied; a negative interaction means the opposite. Neither sign alone proves that query representation is the dominant failure source. This implementation supplies the measurement design and synthetic validation, not paper-ready benchmark findings.
