# SIGIR 2027 submission plan

Checked 9 October 2026 against https://sigir2027.org/pages/submit-full.html.
The official page currently labels the schedule PROPOSED (AoE): abstract January
14, 2027; full paper January 21, 2027. An abstract registration is mandatory.
Treat January 13 as the internal abstract deadline and January 20 as the internal
paper deadline. Recheck the official page before registration.

The submission is anonymous, in English, PDF, at most 9 pages excluding references,
using ACM two-column sigconf format. CCS concepts and keywords are required.
All authors need complete OpenReview profiles; approval can take two weeks.
Finalize the complete author list before abstract submission: the page prohibits
later authorship changes. Use an anonymous artifact URL in the submitted paper,
not this identifying GitHub URL. Concurrent conference/journal submissions are
prohibited. The submission portal link is not yet supplied on the page.

## Work backwards from the abstract

- October 9–31: freeze protocol, audit budgets, obtain three datasets, create
  complete oracle/non-oracle query files and reproduce baselines.
- November 1–30: run BM25, Contriever and a modern embedder on the same question
  sets; perform K/context-budget sweeps and investigate retriever interactions.
- December 1–15: fixed-reader answer evaluation, paired confidence intervals,
  failure taxonomy and related-work verification. Resolve all author profiles.
- December 16–31: freeze main tables, draft the full paper and abstract with
  measured numbers; review novelty claims against the closest studies.
- January 1–10: full draft review, reproduce key results and finalize authors.
- January 13: internal abstract registration target, with results-backed claims.
- January 14: proposed official abstract deadline (AoE).
- January 15–20: final paper checks, anonymization and artifact audit.
- January 21: proposed official full-paper deadline (AoE).

The contribution should explain *when* query representation helps, with a
factorial interaction analysis. Do not promise gains across retrievers before
measuring them. The current qualitative abstract is a working draft; insert
actual dataset sizes, absolute results and paired effect intervals after final runs.
The evaluation topic area explicitly welcomes multi-step IR evaluation; it is
an appropriate framing for this measurement paper.
