"""
wrapper around re-arc. the implementation of this is really jank.
"""

from arc.external import github
from arc import settings
import os
from arc.utils.os_utils import working_dir
from loguru import logger
from datasets import Dataset
import json
import glob

_REPO_NAME = "michaelhodel/re-arc"
_TMP_REPO_DIR = os.path.join(settings.TEMP_ROOT_DIR, "external", "re_arc")
_TMP_DATA_DIR = os.path.join(settings.TEMP_ROOT_DIR, "data", "re_arc", "raw")

# [NOTE][TODO] this is jank af
if not os.path.exists(_TMP_REPO_DIR):
    os.makedirs(_TMP_REPO_DIR)
    github.clone_repo(_REPO_NAME, _TMP_REPO_DIR)

if not os.path.exists(_TMP_DATA_DIR):
    os.makedirs(_TMP_DATA_DIR)


def _raw_data_dir(
    seed: int,
    n_examples: int,
    diff_lb: float,
    diff_ub: float,
) -> str:
    return os.path.join(
        _TMP_DATA_DIR, f"seed-{seed}_n-{n_examples}_diff_lb-{diff_lb}_diff_lb-{diff_ub}"
    )


def _load_dataset(dir) -> Dataset:
    all_data = []

    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(dir, "*.json"))

    for file_path in json_files:
        with open(file_path, "r") as f:
            # Load the JSON content (list of dicts)
            examples = json.load(f)

            # Create a single entry with filename and examples
            entry = {
                "task_id": os.path.basename(file_path).strip(".json"),
                "examples": examples,
            }
            all_data.append(entry)
    dataset = Dataset.from_list(all_data)
    return dataset


def generate_dataset(
    seed: int,
    n_examples: int,
    diff_lb: float,
    diff_ub: float,
    force_recreate: bool = False,
):
    raw_data_dir = _raw_data_dir(seed, n_examples, diff_lb, diff_ub)
    if not os.path.exists(raw_data_dir) or force_recreate:
        with working_dir(_TMP_REPO_DIR):
            import main

            logger.debug(f"generating re_arc dataset to {raw_data_dir}")

            main.generate_dataset(
                path=raw_data_dir,
                seed=seed,
                n_examples=n_examples,
                diff_ub=diff_ub,
                diff_lb=diff_lb,
            )
    return _load_dataset(os.path.join(raw_data_dir, "tasks"))
