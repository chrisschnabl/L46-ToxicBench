import numpy as np
from datasets import DatasetDict

def preprocess_toxicchat(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["user_input"], truncation=True, padding=True)
    tokenized_inputs["label"] = examples["toxicity"]
    return tokenized_inputs

def sample_dataset(dataset, sample_size, seed=1337):
    np.random.seed(seed)
    if len(dataset) < sample_size:
        raise ValueError("Sample size exceeds dataset size!")
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    return dataset.select(indices)
    
def preprocess_realtoxicityprompts(examples, tokenizer, threshold = 0.7):
    texts = []
    labels = []
    
    # Process each example in the batch
    for prompt in examples["prompt"]:
        text = prompt.get("text", "")
        toxicity = prompt.get("toxicity", 0.0)
        label = 1 if toxicity and toxicity > threshold else 0
        texts.append(text)
        labels.append(label)
    
    # Tokenize texts
    tokenized_inputs = tokenizer(texts, truncation=True, padding=True)
    tokenized_inputs["label"] = labels
    return tokenized_inputs

def preprocess_jigsawtoxicity(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["comment_text"], truncation=True, padding=True)
    tokenized_inputs["label"] = examples["toxic"]
    return tokenized_inputs

def preprocess_civilcomments(examples, tokenizer):
    toxicity = examples.get("toxicity", [0.0])
    texts = examples.get("text", [""])
    
    tokenized_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    
    tokenized_inputs["label"] = (toxicity > 0.7).astype(int) if isinstance(toxicity, np.ndarray) else [1 if score > 0.7 else 0 for score in toxicity]
    
    return tokenized_inputs

datasets_config = [
    {"name": "lmsys/toxic-chat", "subset": "toxicchat0124", "preprocess_function": preprocess_toxicchat},
    {"name": "allenai/real-toxicity-prompts", "split": "train", "preprocess_function": preprocess_realtoxicityprompts},
    {"name": "tasksource/jigsaw_toxicity", "split": "train", "preprocess_function": preprocess_jigsawtoxicity},
    {"name": "google/civil_comments", "split": "train", "preprocess_function": preprocess_civilcomments},
]



def split_and_subset(dataset, seed=1337, fraction=1/3, test_size=0.2):
    """
    Splits the dataset into train and test sets, then creates subsets.
    Returns:
        DatasetDict: Tokenized and subsetted train and test splits.
    """
    train_test_split = dataset.train_test_split(test_size=test_size, seed=seed)
    
    if "test" in dataset:
        train_test_split["test"] = dataset["test"]

    np.random.seed(seed)
    subset_dict = {}
    for split in train_test_split.keys():
        split_data = train_test_split[split]
        subset_size = int(len(split_data) * fraction)
        subset_indices = np.random.choice(len(split_data), subset_size, replace=False)
        subset_dict[split] = split_data.select(subset_indices)
    
    return DatasetDict(subset_dict)