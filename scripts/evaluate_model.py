import arckit
from arc.tasks.prompts import puzzlesolver_role_prompt, create_training_prompt
import numpy as np
import re
from typing import Optional, Any


def extract_code(llm_response: str) -> Optional[dict[str, Any]]:
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
    code = code_match.group(1).strip()

    # Create a new dictionary to store the local namespace
    namespace = {}

    # Execute the code in the new namespace
    try:
        exec(code, globals(), namespace)
        return namespace
    except Exception:
        return None


def call_the_finetuned_solver(prompt: str, system_prompt: str) -> str:
    return "Let's solve this puzzle using Python code with the common library functions. We'll first reason about the problem and then write the code to solve it. The `transform` function will take the input grid and return the output grid. Here is the Python code with the comments describing how to solve the problem:\n```python\nfrom common import *\n\nimport numpy as np\nfrom typing import *\n\n# concepts:\n# color merging, border contact, horizontal/vertical connections\n\n# description:\n# In the input you will see horizontal and vertical bars of different colors.\n# If two bars of the same color are touching each other (i.e., they share a border), merge them into one bar of that color.\n# The output will show the final configuration of bars after merging.\n\ndef transform(input_grid: np.ndarray) -> np.ndarray:\n output_grid = np.zeros_like(input_grid)\n\n # Find the connected components (bars) in the input grid\n bars = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)\n\n # Process each bar to check for merging\n for bar in bars:\n color = np.unique(bar)[1] # Get the color of the bar\n # Check if it should merge with any neighboring bars\n for other_bar in bars:\n if bar is other_bar:\n continue # Skip self-comparison\n # Check contact and color match\n if contact(object1=bar, object2=other_bar, connectivity=4):\n # Merge the bars by coloring the output grid with the given color\n blit_object(output_grid, bar, background=Color.BLACK)\n blit_object(output_grid, other_bar, background=Color.BLACK)\n \n return output_grid\n```\n"


train_set, eval_set = arckit.load_data()

results = []

for task in eval_set:
    llm_response = call_the_finetuned_solver(
        prompt=create_training_prompt(task), system_prompt=puzzlesolver_role_prompt
    )

    solution_namespace = extract_code(llm_response)

    test_input = task.test[0][0]

    if solution_namespace:
        if "transform" in solution_namespace:
            try:
                solution = solution_namespace["transform"](test_input)
            except Exception:
                print(f"{task.id} transform function failed to evaluate")
                solution = None
        else:
            print(f"{task.id} no transform function")
            solution = None
    else:
        print(f"{task.id} llm response could not be parsed")
        solution = None

    solution = (
        solution or test_input
    )  # fall back if parsing and transform function fail
    results.append((task.id, np.array_equal(solution, task.test[0][1])))

print(f"{sum([solved for _,solved in results])}/{len(eval_set)} solved")
