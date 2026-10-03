from evidence import sentence_windows
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np


TOP_K = 5
NUM_EXAMPLES = 100
WINDOW_SIZE = 3
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def cosine_top_k(query_vector, matrix, k):
    scores = matrix @ query_vector
    return np.argsort(scores)[::-1][:k]


def count_words(items):
    return sum(len(item["text"].split()) for item in items)


print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=f"validation[:{NUM_EXAMPLES}]",
)

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

passage_recalls = []
window_recalls = []

passage_word_counts = []
window_word_counts = []

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
    windows = []

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

        for window in sentence_windows(group, WINDOW_SIZE):
            windows.append(
                {
                    "title": title,
                    "sentence_ids": window["sentence_ids"],
                    "text": f"{title}. {window['text']}",
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

    window_embeddings = model.encode(
        [item["text"] for item in windows],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    passage_indices = cosine_top_k(
        question_embedding,
        passage_embeddings,
        min(TOP_K, len(passages)),
    )

    window_indices = cosine_top_k(
        question_embedding,
        window_embeddings,
        min(TOP_K, len(windows)),
    )

    retrieved_passages = [
        passages[index]
        for index in passage_indices
    ]

    retrieved_windows = [
        windows[index]
        for index in window_indices
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

    window_hits = set()

    for gold_title, gold_sentence_id in gold_facts:
        for window in retrieved_windows:
            same_title = window["title"] == gold_title
            sentence_inside_window = (
                gold_sentence_id in window["sentence_ids"]
            )

            if same_title and sentence_inside_window:
                window_hits.add((gold_title, gold_sentence_id))
                break

    passage_recall = len(passage_hits) / len(gold_facts)
    window_recall = len(window_hits) / len(gold_facts)

    passage_recalls.append(passage_recall)
    window_recalls.append(window_recall)

    passage_word_counts.append(
        count_words(retrieved_passages)
    )

    window_word_counts.append(
        count_words(retrieved_windows)
    )

    if example_number % 10 == 0:
        print(f"Processed {example_number}/{NUM_EXAMPLES}")


print("\nRESULTS")
print(
    f"Passage supporting-fact recall: "
    f"{np.mean(passage_recalls):.3f}"
)
print(
    f"{WINDOW_SIZE}-sentence window supporting-fact recall: "
    f"{np.mean(window_recalls):.3f}"
)
print(
    f"Passage average retrieved words: "
    f"{np.mean(passage_word_counts):.1f}"
)
print(
    f"{WINDOW_SIZE}-sentence window average retrieved words: "
    f"{np.mean(window_word_counts):.1f}"
)