import arckit
from arc.tasks.prompts import puzzlesolver_role_prompt, create_training_prompt


def call_the_finetuned_solver(prompt: str, system_prompt: str) -> str:
    return ""


train_set, eval_set = arckit.load_data()

for task in eval_set:
    call_the_finetuned_solver(
        prompt=create_training_prompt(task), system_prompt=puzzlesolver_role_prompt
    )

# parse LLM output into function
# evaluate function on test input
# submit two guesses
# score
