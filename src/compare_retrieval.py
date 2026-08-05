from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np


TOP_K = 5
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def cosine_top_k(query_vector, matrix, k):
    scores = matrix @ query_vector
    return np.argsort(scores)[::-1][:k]


print("Loading HotpotQA...")
dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="validation[:100]",
)

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

example = dataset[0]

question = example["question"]
titles = example["context"]["title"]
sentence_groups = example["context"]["sentences"]

passages = []
sentences = []

for title, group in zip(titles, sentence_groups):
    cleaned = [sentence.strip() for sentence in group if sentence.strip()]

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
)

sentence_embeddings = model.encode(
    [item["text"] for item in sentences],
    normalize_embeddings=True,
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

print("\nQUESTION")
print(question)

print("\nGOLD SUPPORTING FACTS")
print(example["supporting_facts"])

print("\nTOP PASSAGES")
for rank, index in enumerate(passage_indices, start=1):
    item = passages[index]
    print(f"{rank}. [{item['title']}] {item['text']}")

print("\nTOP SENTENCES")
for rank, index in enumerate(sentence_indices, start=1):
    item = sentences[index]
    print(
        f"{rank}. [{item['title']} | sentence {item['sentence_id']}] "
        f"{item['text']}"
    )