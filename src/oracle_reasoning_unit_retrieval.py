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
        "Carlo Rovelli worked in France since",
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

    passages = []

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

    passage_embeddings = model.encode(
        [item["text"] for item in passages],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Baseline
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

    # Oracle reasoning-unit retrieval
    oracle_units = ORACLE_UNITS[question]

    oracle_embeddings = model.encode(
        oracle_units,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    oracle_scores = oracle_embeddings @ passage_embeddings.T
    combined_scores = np.max(oracle_scores, axis=0)

    oracle_indices = np.argsort(
        combined_scores
    )[::-1][: min(TOP_K, len(passages))]

    oracle_passages = [
        passages[index]
        for index in oracle_indices
    ]

    baseline_titles = {
        item["title"]
        for item in baseline_passages
    }

    oracle_titles = {
        item["title"]
        for item in oracle_passages
    }

    baseline_hits = {
        fact
        for fact in gold_facts
        if fact[0] in baseline_titles
    }

    oracle_hits = {
        fact
        for fact in gold_facts
        if fact[0] in oracle_titles
    }

    baseline_recall = len(baseline_hits) / len(gold_facts)
    oracle_recall = len(oracle_hits) / len(gold_facts)

    baseline_recalls.append(baseline_recall)
    oracle_recalls.append(oracle_recall)

    baseline_word_counts.append(
        count_words(baseline_passages)
    )

    oracle_word_counts.append(
        count_words(oracle_passages)
    )

    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print(f"Oracle units: {oracle_units}")
    print(f"Baseline recall: {baseline_recall:.3f}")
    print(f"Oracle recall: {oracle_recall:.3f}")
    print(
        "Baseline titles:",
        [item["title"] for item in baseline_passages],
    )
    print(
        "Oracle titles:",
        [item["title"] for item in oracle_passages],
    )


print("\nFINAL RESULTS")
print(
    f"Baseline recall: "
    f"{np.mean(baseline_recalls):.3f}"
)
print(
    f"Oracle reasoning-unit recall: "
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