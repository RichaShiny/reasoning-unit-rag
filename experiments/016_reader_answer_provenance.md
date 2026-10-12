# Bind answer evaluation to declared reader requests

An answer indexed only by question ID and condition can be accidentally reused after context, prompt or model changes. The diagnostic matrix now requires a versioned answer bundle containing a request hash for every prediction. Legacy maps of bare answer strings are rejected.

## Fixed reader configuration

Create `reader.json` using the actual model identifier, immutable revision or hosted snapshot identifier, prompt and complete generation settings. For example (replace the model/revision placeholders):

```json
{
  "model": "MODEL_ID",
  "revision": "IMMUTABLE_REVISION",
  "prompt_template": "Answer using only this context.\n{context}\nQuestion: {question}",
  "decoding": {"temperature": 0, "max_new_tokens": 128}
}
```

Only plain `{question}` and `{context}` placeholders are allowed; both are required. Escaped literal braces are supported. The configuration describes a single rendered text prompt. If an adapter adds chat wrappers, truncation or other preprocessing, preserve those settings/revisions as part of the declared configuration's JSON decoding settings and generation records. Do not claim equivalent requests when the actual generation pipeline differs.

The validator checks structure, not whether a revision resolves to an immutable model. Record all effective settings rather than rely on undocumented defaults. The evaluator does not load the reader or make inference requests.

## Export and return answers

Run retrieval with the fixed configuration:

```sh
python3 src/diagnostic_matrix.py --dataset hotpotqa \
  --input results/replication-100/examples.json --retriever bm25 \
  --top-k 5 --context-budget 512 --reader-config reader.json \
  --output results/reader-retrieval
python3 scripts/export_reader_requests.py results/reader-retrieval/predictions.jsonl \
  --reader-config reader.json --output reader-requests.json
```

The exporter requires a completed run with the expected question IDs/count, consistent conditions and metric contract, and intact reader input hashes. It verifies that serialized context equals the retrieved texts. It exports the original question, retrieved context, rendered prompt, configuration and hashes. It excludes gold labels, gold answers, and oracle query text. Retrieved context may naturally contain the answer.

For each `requests[example_id][condition]`, run the declared reader on its exact `prompt`. Preserve the returned output and `reader_request_sha256`. Assemble this envelope with actual outputs (the values below illustrate the format only):

```json
{
  "schema_version": 1,
  "reader_config": {"model": "MODEL_ID", "revision": "IMMUTABLE_REVISION", "prompt_template": "{context}\n{question}", "decoding": {"temperature": 0}},
  "predictions": {
    "QUESTION_ID": {
      "sentence/original": {"answer": "ACTUAL_READER_OUTPUT", "reader_request_sha256": "COPY_THE_EXPORTED_64_CHARACTER_HASH"}
    }
  }
}
```

Copy the configuration from the export exactly and include every evaluated question and cell. The illustrative single-cell envelope is not a complete matrix result. Empty answer strings are allowed as genuine reader outputs/abstentions. Missing or extra questions/cells fail before retrieval scoring.

Rerun the same retrieval settings in a new directory with `--reader-config reader.json --answers answers.json`. Every answer hash must match the newly computed request before EM/F1 is scored. Changing context order, whitespace, question, model, revision, prompt or settings invalidates the old answer when it changes the request. If some examples were already written before a mismatch is discovered, the run is marked failed and no summary is produced; never analyze those partial predictions as a completed experiment.

A retrieval-only run can also be exported with a reader configuration later. If a run already declares a reader configuration, export requires that same configuration. Unbound legacy retrieval records must be regenerated to obtain input hashes.

## Hashes and limits

SHA-256 uses canonical JSON with sorted keys, compact separators, UTF-8 text and no nonfinite numbers. Input hashes bind question/context; config hashes bind the full reader configuration; request hashes bind both plus the rendered prompt. A changed request with the same question ID cannot reuse an old prediction. Summaries reject mixed reader configurations and unbound answer metrics.

These hashes detect accidental mismatches. They do not attest that an external provider executed the declared model or that the submitted prediction came from inference. Preserve raw provider outputs and execution metadata separately. No benchmark answer results are introduced by this implementation; round-trip tests use explicit synthetic fixtures.
