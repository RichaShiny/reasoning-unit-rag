from pathlib import Path
import json

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


TOP_K = 5
NUM_EXAMPLES = 100
NUM_FAILURES_TO_SAVE = 20
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_PATH = Path("results/failure_analysis.json")


def cosine_top_k(
    query_vector: np.ndarray,
    matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    scores = matrix @ query_vector
    return np.argsort(scores)[::-1][:k]


print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=f"validation[:{NUM_EXAMPLES}]",
)

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

failure_cases = []

for example_number, example in enumerate(dataset, start=1):
    question = example["question"]
    titles = example["context"]["title"]
    sentence_groups = example["context"]["sentences"]

    gold_facts = set(
        zip(
            example["supporting_facts"]["title"],
            example["supporting_facts"]["sent_id"],
        )
    )

    passages = []
    sentences = []

    for title, group in zip(titles, sentence_groups):
        cleaned = [
            sentence.strip()
            for sentence in group
            if sentence and sentence.strip()
        ]

        passages.append(
            {
                "title": title,
                "text": f"{title}. {' '.join(cleaned)}",
            }
        )

        for sentence_id, sentence in enumerate(cleaned):
            sentences.append(
                {
                    "title": title,
                    "sentence_id": sentence_id,
                    "text": f"{title}. {sentence}",
                }
            )

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    passage_embeddings = model.encode(
        [item["text"] for item in passages],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sentence_embeddings = model.encode(
        [item["text"] for item in sentences],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    passage_indices = cosine_top_k(
        question_embedding,
        passage_embeddings,
        min(TOP_K, len(passages)),
    )

    sentence_indices = cosine_top_k(
        question_embedding,
        sentence_embeddings,
        min(TOP_K, len(sentences)),
    )

    retrieved_passages = [
        passages[index]
        for index in passage_indices
    ]

    retrieved_sentences = [
        sentences[index]
        for index in sentence_indices
    ]

    retrieved_passage_titles = {
        item["title"]
        for item in retrieved_passages
    }

    passage_hits = {
        fact
        for fact in gold_facts
        if fact[0] in retrieved_passage_titles
    }

    retrieved_sentence_facts = {
        (item["title"], item["sentence_id"])
        for item in retrieved_sentences
    }

    sentence_hits = gold_facts & retrieved_sentence_facts

    passage_recall = len(passage_hits) / len(gold_facts)
    sentence_recall = len(sentence_hits) / len(gold_facts)

    passage_succeeded = passage_recall == 1.0
    sentence_failed = sentence_recall < 1.0

    if passage_succeeded and sentence_failed:
        missing_gold_facts = sorted(
            gold_facts - sentence_hits
        )

        failure_cases.append(
            {
                "example_id": example["id"],
                "question": question,
                "answer": example["answer"],
                "question_type": example["type"],
                "difficulty": example["level"],
                "gold_facts": [
                    {
                        "title": title,
                        "sentence_id": sentence_id,
                    }
                    for title, sentence_id in sorted(gold_facts)
                ],
                "missing_gold_facts": [
                    {
                        "title": title,
                        "sentence_id": sentence_id,
                    }
                    for title, sentence_id in missing_gold_facts
                ],
                "passage_recall": passage_recall,
                "sentence_recall": sentence_recall,
                "retrieved_passage_titles": [
                    item["title"]
                    for item in retrieved_passages
                ],
                "retrieved_sentences": retrieved_sentences,
            }
        )

    if example_number % 10 == 0:
        print(f"Processed {example_number}/{NUM_EXAMPLES}")


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

selected_failures = failure_cases[:NUM_FAILURES_TO_SAVE]

with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
    json.dump(
        selected_failures,
        output_file,
        indent=2,
        ensure_ascii=False,
    )


print("\nFAILURE ANALYSIS")
print(
    "Cases where passage retrieval found all gold pages "
    "but sentence retrieval missed evidence:"
)
print(f"{len(failure_cases)}/{NUM_EXAMPLES}")

print(
    f"\nSaved the first {len(selected_failures)} cases to "
    f"{OUTPUT_PATH}"
)

for case_number, case in enumerate(
    selected_failures[:5],
    start=1,
):
    print(f"\n{'=' * 70}")
    print(f"FAILURE CASE {case_number}")
    print(f"Question: {case['question']}")
    print(f"Answer: {case['answer']}")
    print(
        "Missing gold facts: "
        f"{case['missing_gold_facts']}"
    )

    print("\nRetrieved sentences:")

    for rank, sentence in enumerate(
        case["retrieved_sentences"],
        start=1,
    ):
        print(
            f"{rank}. "
            f"[{sentence['title']} | "
            f"sentence {sentence['sentence_id']}] "
            f"{sentence['text']}"
        )