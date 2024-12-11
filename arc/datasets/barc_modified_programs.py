from datasets import load_dataset

HF_DATASET_NAME = "minhxle/barc-induction-modified-programs-2k"


def get_raw_dataset():
    return load_dataset(HF_DATASET_NAME)
