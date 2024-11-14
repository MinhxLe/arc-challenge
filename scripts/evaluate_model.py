import arckit
from arc.tasks.prompts import puzzlesolver_role_prompt, create_training_prompt
import numpy as np
import re
from typing import Optional
from datasets import load_dataset
import os
import uuid
import importlib

TMP_SOLUTION_DIR = "data/tmp_solutions"

# only used for mocking llm responses
dataset = load_dataset(
    "barc0/induction_100k_gpt4o-mini_generated_problems_seed100.jsonl_messages_format_0.3"
)


def create_module(llm_response: str) -> Optional[str]:
    """
    Extracts Python code from an LLM response string and returns a dictionary
    containing the executable objects defined in the code.

    Args:
        llm_response (str): The string containing the LLM response with Python code

    Returns:
        Optional[dict]: Dictionary containing the defined objects, or None if no code found
    """
    # Extract code between Python code blocks
    code_match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)

    if not code_match:
        return None

    # Extract the code content
    code = (
        code_match.group(1)
        .strip()
        .replace("from common import", "from arc.dsl.common import")
    )

    # Create the temporary directory if it doesn't exist
    os.makedirs(TMP_SOLUTION_DIR, exist_ok=True)

    # Generate a unique filename using UUID
    module_name = f"solution_{uuid.uuid4().hex[:8]}"
    filename = module_name + ".py"
    file_path = os.path.join(TMP_SOLUTION_DIR, filename)

    try:
        # Write the code to the file
        with open(file_path, "w") as f:
            f.write(code)
        return module_name
    except Exception as e:
        print(f"Error writing module: {e}")
        if os.path.exists(file_path):
            os.unlink(file_path)
        return None


def call_the_finetuned_solver(task_index: int, prompt: str, system_prompt: str) -> str:
    return dataset["train_sft"][task_index]["messages"][2]["content"]  # type: ignore


train_set, eval_set = arckit.load_data()

results = []

for i, task in enumerate(eval_set):
    llm_response = call_the_finetuned_solver(
        i, prompt=create_training_prompt(task), system_prompt=puzzlesolver_role_prompt
    )

    solution_module_name = create_module(llm_response)

    test_input = task.test[0][0]

    if solution_module_name:
        try:
            solution_module = importlib.import_module(
                f'{TMP_SOLUTION_DIR.replace("/",".")}.{solution_module_name}'
            )
            try:
                solution = solution_module.transform(test_input)
            except Exception:
                print(f"{task.id} transform failed to execute")
                solution = None

        except Exception:
            print(f"{task.id} solution module failed to import")
            solution = None

        os.unlink(os.path.join(TMP_SOLUTION_DIR, solution_module_name) + ".py")

    else:
        print(f"{task.id} llm response could not be parsed")
        solution = None

    if solution is None:
        solution = test_input  # fall back if parsing and transform function fail
    results.append((task.id, np.array_equal(solution, task.test[0][1])))

print(f"{sum([solved for _,solved in results])}/{len(eval_set)} solved")
