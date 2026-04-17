from datasets import load_dataset, DatasetDict

dataset = load_dataset("ag_news")

dataset_dict = DatasetDict({
    "train": dataset["train"],
    "test": dataset["test"]
})

print(dataset_dict)

def save_dataset(dataset, base_name, split_name, num_examples):
    dataset_dict = DatasetDict({split_name: dataset})
    dataset_name = f"{base_name}_{split_name}_{num_examples}"
    path = f"./data/{dataset_name}"
    dataset_dict.save_to_disk(path)
    print(f"{dataset_name} saved to: {path}")

dataset_base_name = "AgNews"

# ------------------------------
# TRAIN + VAL
# ------------------------------

train = dataset_dict["train"].shuffle(seed=42)

n = 1250
subset = train.select(range(n))

split_dataset = subset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")

train_set = split_dataset["train"]
val_set = split_dataset["test"]

save_dataset(train_set, dataset_base_name, "train", len(train_set))
save_dataset(val_set, dataset_base_name, "val", len(val_set))

# ------------------------------
# TEST
# ------------------------------

test_set = dataset_dict["test"].shuffle(seed=42).select(range(1000))

save_dataset(test_set, dataset_base_name, "test", len(test_set))