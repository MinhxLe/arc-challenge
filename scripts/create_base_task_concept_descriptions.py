from arc.core import Concept
from arc.external import openai
from arc.tasks import prompts
import csv


def generate_puzzle(concept: Concept):
    return openai.complete_structured(
        prompts.create_base_task_puzzle_descriptions_prompt(concept, 1),
        prompts.PuzzleDescriptions,
        system_prompt=prompts.puzzlemaker_role_prompt,
        temperature=1,
    ).descriptions[0]


# Open the CSV file in write mode
with open("./scripts/data/base_task_descriptions.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["concept", "task_description"])

    # Loop through each concept and write to the CSV
    for concept in Concept:
        writer.writerow([concept.value, generate_puzzle(concept)])
