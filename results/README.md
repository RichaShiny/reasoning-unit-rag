# Baseline run artifacts

`replication-100/` and `validation-500/` contain completed runs described in [Experiment 008](../experiments/008_corrected_baselines.md).

- `manifest.json`: exact IDs, pinned dataset/model revisions, configuration, versions, and completion state.
- `examples.json`: exact sampled HotpotQA contexts and evaluator labels.
- `predictions.jsonl`: ranked sentences, query lists, and per-example metrics.
- `summary.json`: aggregate and question-type results.
- `diagnostics.json`: artifact audit and failure categories, regenerable with `python3 scripts/audit_run.py RUN_DIRECTORY`.
- `paired_analysis.json`: seeded paired-bootstrap intervals and input-content hash.

The questions and Wikipedia context/excerpt text are from [HotpotQA (Yang et al., 2018)](https://hotpotqa.github.io/), via [the pinned Hugging Face dataset](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab). The dataset declares [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); that license applies to reproduced dataset content and derived excerpts here. The project's MIT code license does not replace the dataset license. Source text is preserved in the sample; retrieved text is stripped and prefixed with its article title, and prediction/diagnostic annotations were added by this project.

`local-pilot-5/` and `local-pilot-5-replay/` record the actual local generator diagnostic and cache-only replay described in [Experiment 010](../experiments/010_local_generator_pilot.md). The five questions are a subset of the inspected replication sample. All local responses failed the structured-query contract, so the automated method used original-question fallback for every example. `local-pilot-5/generator-cache/` preserves the exact requests/raw outputs for replay; model weights are not included. `replay_verification.json` records same-machine ID/response/query/metric/ranking checks and score tolerance.
