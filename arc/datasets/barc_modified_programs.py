from arckit.data import Task
from datasets.dataset_dict import DatasetDict
from datasets.exceptions import DatasetNotFoundError
from arc.external import huggingface
from arc.tasks import prompts
from datasets import load_dataset, Dataset
from arc.program import Program, remove_comments, ProgramExecution
from loguru import logger

HF_DATASET_NAME = "minhxle/barc-induction-modified-programs-2k"
HF_FINETUNE_DATASET_NAME = "minhxle/barc-induction-modified-programs-2k-convo-style"


def get_raw_dataset():
    return load_dataset(HF_DATASET_NAME)


def get_finetune_dataset():
    try:
        return load_dataset(HF_FINETUNE_DATASET_NAME)
    except DatasetNotFoundError:
        logger.debug("generating dataset")

    raw_dataset = get_raw_dataset()["train"]
    all_messages = []
    for r in raw_dataset:
        task = Task(**r["task"])
        initial_program = Program.from_source(
            remove_comments(r["modified_program_source"])
        )
        execution = ProgramExecution(initial_program, task)
        original_program_source = r["original_program_source"]
        messages = [
            dict(role="system", content=prompts.programmer_role_prompt),
            dict(
                role="user",
                content=prompts.create_improve_solve_task_prompt(task, [execution]),
            ),
            dict(
                role="assistant",
                content=prompts.create_python_source_string(original_program_source),
            ),
        ]
        all_messages.append(dict(messages=messages))
        dataset = Dataset.from_list(all_messages)
        dataset_dict = DatasetDict(train=dataset)
        huggingface.upload_dataset(HF_FINETUNE_DATASET_NAME, dataset_dict)
        return dataset_dict
