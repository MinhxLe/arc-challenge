from arc.datasets.barc_generated_problems import get_dataset, extract_generated_task
from loguru import logger
from multiprocessing import Pool
from functools import partial


def process_row(row, index):
    try:
        return extract_generated_task(row["messages"])
    except Exception as e:
        logger.error(f"error on {index}: {e}")
        return None


def main():
    dataset = get_dataset()

    # Create a pool of workers
    with Pool() as pool:
        # Create list of (row, index) tuples
        items = [(row, i) for i, row in enumerate(dataset["train_sft"])]
        # Map the processing function to all items
        tasks = pool.starmap(process_row, items)
        # Filter out None values from errors
        tasks = [t for t in tasks if t is not None]


if __name__ == "__main__":
    main()
