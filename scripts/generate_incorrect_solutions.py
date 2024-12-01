from numpy import diff
from arc.datasets.barc_generated_problems import get_parsed_dataset, GeneratedTask
from arc.external import openai
from arc.tasks import prompts


tasks = get_parsed_dataset()


def generate_modified_code(task: GeneratedTask, difficulty: int):
    assert 1 <= difficulty <= 10
    return openai.complete(
        system_prompt=f"""{prompts.programmer_role_prompt}
You are given a problem specification and the source code for a solution for the program. Your intention is to modify the source code such that the program no longer satisfy the specification. The changed program is intended to be given to a student programmer so that they will be able to revert it. Given on a difficulty score of 1 to 10, you should output the modified program such that a program of score 1 will be easy to debug and 10 will most challenging/requires most change.

Changes should
1. change the program behavior.
2. NOT be variable renaming.
3. Change the output type.
""",
        prompt=f"""Description:
{task.description}

Original source code:
```python
{task.program.source}

Output a modified program to debug with a difficulty of {difficulty}.
```""",
        return_raw=True,
    )


resp = generate_modified_code(tasks[0], 1)
resp = generate_modified_code(tasks[0], 1)
resp2 = generate_modified_code(tasks[0], 10)
