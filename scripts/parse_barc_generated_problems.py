from arc.datasets.barc_generated_problems import (
    get_dataset,
    extract_generated_task,
    GeneratedTask,
)
from loguru import logger
import pickle as pkl

dataset = get_dataset()

tasks = []
# parallelize this with multiprocessing
for i, row in enumerate(dataset["train_sft"]):
    try:
        tasks.append(extract_generated_task(row["messages"]))
    except Exception as e:
        logger.error(f"error on {i}: {e}")

# TODO move this into library
# 97185/97238 managed to be parsed
# save tasks down with pickle to tmp.pkl
serialized_tasks = [GeneratedTask.serialize(t) for t in tasks]

with open("tmp/processed/train_barc_generated_problems.pkl", "wb") as f:
    pkl.dump(serialized_tasks, f)


with open("tmp/processed/train_barc_generated_problems.pkl", "rb") as f:
    raw = pkl.load(f)
    read_tasks = [GeneratedTask.deserialize(x) for x in raw]
