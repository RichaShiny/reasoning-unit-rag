from evidence import indexed_sentences
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np


TOP_K = 5
NUM_EXAMPLES = 100
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
sentence_recalls = []

passage_word_counts = []
sentence_word_counts = []

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

        for sentence_id, sentence in indexed_sentences(group):
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

    retrieved_passages = [passages[index] for index in passage_indices]
    retrieved_sentences = [sentences[index] for index in sentence_indices]

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

    passage_recalls.append(passage_recall)
    sentence_recalls.append(sentence_recall)

    passage_word_counts.append(count_words(retrieved_passages))
    sentence_word_counts.append(count_words(retrieved_sentences))

    if example_number % 10 == 0:
        print(f"Processed {example_number}/{NUM_EXAMPLES}")


print("\nRESULTS")
print(f"Passage supporting-fact recall: {np.mean(passage_recalls):.3f}")
print(f"Sentence supporting-fact recall: {np.mean(sentence_recalls):.3f}")
print(f"Passage average retrieved words: {np.mean(passage_word_counts):.1f}")
print(f"Sentence average retrieved words: {np.mean(sentence_word_counts):.1f}")