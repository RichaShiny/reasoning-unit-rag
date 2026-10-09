# Benchmark validation before normalization

The diagnostic matrix now rejects malformed benchmark rows before constructing candidates or gold sets. In columnar HotpotQA and 2Wiki inputs, unequal title/sentence or title/sentence-ID arrays previously passed through `zip`, silently discarding documents or gold evidence. Both directions of a length mismatch now raise `ValueError`.

Native pair lists and columnar inputs remain supported. Sentence IDs must be nonnegative integers (not booleans, strings or floats), and sentence groups must contain strings. Empty sentence strings remain valid placeholders: their native coordinates are preserved, and a gold label pointing at an empty sentence remains an error.

MuSiQue checks cover paragraph types, nonempty paragraph text, boolean support flags, unique positive decomposition step IDs and valid dependencies. Duplicate document IDs, absent gold support and unsupported dataset names fail explicitly. No paragraph labels are promoted into sentence-level annotations by this validation change; MuSiQue's existing paragraph-proxy metrics still require separate interpretation.

Validation: `python -m unittest discover -s tests -v`. The regression suite exercises both native and columnar inputs, discarded-label cases, malformed coordinates, empty sentence placeholders and MuSiQue dependencies. This change does not rerun benchmark experiments or establish new research results.
