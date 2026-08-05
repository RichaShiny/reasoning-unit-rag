from datasets import load_dataset

print("Downloading HotpotQA sample...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="validation[:100]",
)

print(dataset[0])
