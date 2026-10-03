"""Keep HotpotQA evidence coordinates while excluding empty retrieval text."""


def indexed_sentences(group):
    """Return (original sentence ID, stripped text), never renumbering IDs."""
    return [
        (sentence_id, sentence.strip())
        for sentence_id, sentence in enumerate(group)
        if sentence and sentence.strip()
    ]


def sentence_windows(group, size=3):
    """Build windows over nonempty sentences with explicit source coordinates."""
    if size < 1 or size % 2 == 0:
        raise ValueError("Window size must be a positive odd integer")
    indexed = indexed_sentences(group)
    half = size // 2
    for center in range(len(indexed)):
        members = indexed[max(0, center - half):center + half + 1]
        yield {
            "sentence_ids": [sentence_id for sentence_id, _ in members],
            "text": " ".join(text for _, text in members),
        }
