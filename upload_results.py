import pickle
import pandas as pd
from datasets import DatasetDict, Dataset
from huggingface_hub import login
import os

def parse_file_name(file_name):
    """Parse dataset and model names from file name."""
    if "_" in file_name:
        dataset_name, model_name = file_name.split("_", 1)
    elif "-" in file_name:
        dataset_name, model_name = file_name.split("-", 1)
    else:
        dataset_name, model_name = "unknown", file_name

    model_name = model_name.split(".")[0]
    return dataset_name, model_name

def load_metrics(
    pickle_directory: str,
    *,
    dict_of_datasets: bool = False,
    force_dataset_name: str = None
) -> pd.DataFrame:
    """
    Generalized loader that:
      - Reads .pkl files in <dataset>_<model>.pkl format.
      - If `dict_of_datasets` is True, each file is expected to be { datasetName: {metric1: val1, ...}, ... }.
        (Equivalent to "Scenario 2" logic.)
      - If `dict_of_datasets` is False, each file is a single dictionary of metrics
        (Equivalent to "Scenario 1" or "Scenario 3" logic).
      - If `force_dataset_name` is given, we override the dataset name (Scenario 3 style).
    """
    grouped_data = []
    pickle_files = sorted(f for f in os.listdir(pickle_directory) if f.endswith(".pkl"))

    for file in pickle_files:
        # Example file name: "datasetName_modelName.pkl"
        dataset_name, model_name = file.split("_", 1)
        model_name = model_name.replace(".pkl", "")

        file_path = os.path.join(pickle_directory, file)
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        if dict_of_datasets:
            # SCENARIO 2: data is {someDatasetName: { ...metrics... }, anotherDataset: { ... } }
            if not isinstance(data, dict):
                print(f"Warning: Expected dictionary-of-datasets, but got type {type(data)} for {file}. Skipping.")
                continue

            for ds_name, metric_vals in data.items():
                # metric_vals should itself be a dict of metrics
                grouped_data.append({
                    "dataset": ds_name,
                    "model": model_name,
                    **metric_vals
                })
        else:
            # SCENARIO 1 or 3: data is a single dict of metrics
            # Overwrite dataset_name if a "forced" dataset name is provided
            final_ds_name = force_dataset_name if force_dataset_name else dataset_name

            if not isinstance(data, dict):
                print(f"Warning: Expected dict of metrics, but got type {type(data)} for {file}. Skipping.")
                continue

            grouped_data.append({
                "dataset": final_ds_name,
                "model": model_name,
                **data
            })

    return pd.DataFrame(grouped_data)

def create_and_push_dataset(grouped_data, hf_username, hf_repo_name, local_path):
    """Create a DatasetDict and push it to Hugging Face Hub."""
    df = pd.DataFrame(grouped_data)
    hf_dataset = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({"metrics": hf_dataset})

    # Save locally for backup
    dataset_dict.save_to_disk(local_path)

    # Push to Hugging Face Hub
    repo_name = f"{hf_username}/{hf_repo_name}"
    dataset_dict.push_to_hub(repo_name)

    print(f"Dataset successfully pushed to {repo_name} on Hugging Face Hub.")
