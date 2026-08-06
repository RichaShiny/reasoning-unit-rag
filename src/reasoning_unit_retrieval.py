from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

from reasoning_units import generate_reasoning_units


TOP_K = 5
NUM_EXAMPLES = 100
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def cosine_top_k(
    query_vector: np.ndarray,
    matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    scores = matrix @ query_vector
    return np.argsort(scores)[::-1][:k]


def count_words(items: list[dict]) -> int:
    return sum(len(item["text"].split()) for item in items)


print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=f"validation[:{NUM_EXAMPLES}]",
)

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

baseline_recalls = []
reasoning_unit_recalls = []

baseline_word_counts = []
reasoning_unit_word_counts = []

reasoning_unit_counts = []


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

    for title, group in zip(titles, sentence_groups):
        cleaned_sentences = [
            sentence.strip()
            for sentence in group
            if sentence and sentence.strip()
        ]

        passages.append(
            {
                "title": title,
                "text": f"{title}. {' '.join(cleaned_sentences)}",
            }
        )

    passage_texts = [item["text"] for item in passages]

    passage_embeddings = model.encode(
        passage_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Baseline retrieval: retrieve once using the full question

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    baseline_indices = cosine_top_k(
        question_embedding,
        passage_embeddings,
        min(TOP_K, len(passages)),
    )

    baseline_passages = [
        passages[index]
        for index in baseline_indices
    ]

    # Reasoning-unit retrieval

    reasoning_units = generate_reasoning_units(question)
    reasoning_unit_counts.append(len(reasoning_units))

    reasoning_unit_embeddings = model.encode(
        reasoning_units,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Shape:
    # number of reasoning units × number of passages
    unit_passage_scores = (
        reasoning_unit_embeddings
        @ passage_embeddings.T
    )

    # Give each passage its strongest score across all reasoning units.
    # This lets the units contribute while maintaining the same
    # final Top-K retrieval budget as the baseline.
    combined_scores = np.max(
        unit_passage_scores,
        axis=0,
    )

    reasoning_unit_indices = np.argsort(
        combined_scores
    )[::-1][: min(TOP_K, len(passages))]

    reasoning_unit_passages = [
        passages[index]
        for index in reasoning_unit_indices
    ]

    
    # Evaluation
    
    baseline_titles = {
        item["title"]
        for item in baseline_passages
    }

    reasoning_unit_titles = {
        item["title"]
        for item in reasoning_unit_passages
    }

    baseline_hits = {
        fact
        for fact in gold_facts
        if fact[0] in baseline_titles
    }

    reasoning_unit_hits = {
        fact
        for fact in gold_facts
        if fact[0] in reasoning_unit_titles
    }

    baseline_recall = (
        len(baseline_hits) / len(gold_facts)
    )

    reasoning_unit_recall = (
        len(reasoning_unit_hits) / len(gold_facts)
    )

    baseline_recalls.append(baseline_recall)
    reasoning_unit_recalls.append(
        reasoning_unit_recall
    )

    baseline_word_counts.append(
        count_words(baseline_passages)
    )

    reasoning_unit_word_counts.append(
        count_words(reasoning_unit_passages)
    )

    if example_number % 10 == 0:
        print(
            f"Processed "
            f"{example_number}/{NUM_EXAMPLES}"
        )


print("\nRESULTS")

print(
    "Baseline passage supporting-fact recall: "
    f"{np.mean(baseline_recalls):.3f}"
)

print(
    "Reasoning-unit passage supporting-fact recall: "
    f"{np.mean(reasoning_unit_recalls):.3f}"
)

print(
    "Baseline average retrieved words: "
    f"{np.mean(baseline_word_counts):.1f}"
)

print(
    "Reasoning-unit average retrieved words: "
    f"{np.mean(reasoning_unit_word_counts):.1f}"
)

print(
    "Average reasoning units generated per question: "
    f"{np.mean(reasoning_unit_counts):.2f}"
)

recall_difference = (
    np.mean(reasoning_unit_recalls)
    - np.mean(baseline_recalls)
)

print(
    "Recall difference: "
    f"{recall_difference:+.3f}"
)