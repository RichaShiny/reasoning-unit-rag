from evidence import indexed_sentences
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np


TOP_K = 5
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


ORACLE_UNITS = {
    "The arena where the Lewiston Maineiacs played their home games can seat how many people?": [
        "Lewiston Maineiacs home arena",
        "Androscoggin Bank Colisée seating capacity",
    ],
    "Who is older, Annie Morton or Terry Richardson?": [
        "Annie Morton birth date",
        "Terry Richardson birth date",
    ],
    "Where is the company that Sachin Warrier worked for headquartered?": [
        "Sachin Warrier employer",
        "Tata Consultancy Services headquarters",
    ],
    "Seven Brief Lessons on Physics was written by an Italian physicist that has worked in France since what year?": [
        "Seven Brief Lessons on Physics author",
        "Carlo Rovelli worked in France since what year",
    ],
    "Ralph Hefferline was a psychology professor at a university that is located in what city?": [
        "Ralph Hefferline university",
        "Columbia University location",
    ],
}


def cosine_top_k(query_vector, matrix, k):
    scores = matrix @ query_vector
    return np.argsort(scores)[::-1][:k]


def count_words(items):
    return sum(len(item["text"].split()) for item in items)


print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="validation",
)

selected_examples = [
    example
    for example in dataset
    if example["question"] in ORACLE_UNITS
]

print(f"Found {len(selected_examples)} oracle examples")
print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

baseline_recalls = []
oracle_recalls = []

baseline_word_counts = []
oracle_word_counts = []


for example in selected_examples:
    question = example["question"]
    titles = example["context"]["title"]
    sentence_groups = example["context"]["sentences"]

    gold_facts = set(
        zip(
            example["supporting_facts"]["title"],
            example["supporting_facts"]["sent_id"],
        )
    )

    sentences = []

    for title, group in zip(titles, sentence_groups):
        cleaned = [
            sentence.strip()
            for sentence in group
            if sentence and sentence.strip()
        ]

        for sentence_id, sentence in indexed_sentences(group):
            sentences.append(
                {
                    "title": title,
                    "sentence_id": sentence_id,
                    "text": f"{title}. {sentence}",
                }
            )

    sentence_embeddings = model.encode(
        [item["text"] for item in sentences],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Baseline sentence retrieval
    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    baseline_indices = cosine_top_k(
        question_embedding,
        sentence_embeddings,
        min(TOP_K, len(sentences)),
    )

    baseline_sentences = [
        sentences[index]
        for index in baseline_indices
    ]

    # Oracle reasoning-unit sentence retrieval
    oracle_units = ORACLE_UNITS[question]

    oracle_embeddings = model.encode(
        oracle_units,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    unit_sentence_scores = (
        oracle_embeddings
        @ sentence_embeddings.T
    )

    combined_scores = np.max(
        unit_sentence_scores,
        axis=0,
    )

    oracle_indices = np.argsort(
        combined_scores
    )[::-1][: min(TOP_K, len(sentences))]

    oracle_sentences = [
        sentences[index]
        for index in oracle_indices
    ]

    baseline_facts = {
        (item["title"], item["sentence_id"])
        for item in baseline_sentences
    }

    oracle_facts = {
        (item["title"], item["sentence_id"])
        for item in oracle_sentences
    }

    baseline_hits = gold_facts & baseline_facts
    oracle_hits = gold_facts & oracle_facts

    baseline_recall = (
        len(baseline_hits) / len(gold_facts)
    )

    oracle_recall = (
        len(oracle_hits) / len(gold_facts)
    )

    baseline_recalls.append(baseline_recall)
    oracle_recalls.append(oracle_recall)

    baseline_word_counts.append(
        count_words(baseline_sentences)
    )

    oracle_word_counts.append(
        count_words(oracle_sentences)
    )

    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print(f"Oracle units: {oracle_units}")
    print(f"Gold facts: {sorted(gold_facts)}")
    print(f"Baseline recall: {baseline_recall:.3f}")
    print(f"Oracle recall: {oracle_recall:.3f}")

    print("\nBaseline sentences:")
    for rank, item in enumerate(
        baseline_sentences,
        start=1,
    ):
        print(
            f"{rank}. [{item['title']} | "
            f"sentence {item['sentence_id']}] "
            f"{item['text']}"
        )

    print("\nOracle sentences:")
    for rank, item in enumerate(
        oracle_sentences,
        start=1,
    ):
        print(
            f"{rank}. [{item['title']} | "
            f"sentence {item['sentence_id']}] "
            f"{item['text']}"
        )


print("\nFINAL RESULTS")
print(
    f"Baseline sentence recall: "
    f"{np.mean(baseline_recalls):.3f}"
)
print(
    f"Oracle reasoning-unit sentence recall: "
    f"{np.mean(oracle_recalls):.3f}"
)
print(
    f"Baseline average retrieved words: "
    f"{np.mean(baseline_word_counts):.1f}"
)
print(
    f"Oracle average retrieved words: "
    f"{np.mean(oracle_word_counts):.1f}"
)
print(
    f"Recall difference: "
    f"{np.mean(oracle_recalls) - np.mean(baseline_recalls):+.3f}"
)