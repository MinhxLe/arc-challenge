import random
import string
from arc.tasks import prompts
from arc.core import Concept
from arc.external import openai
from arckit import Task
import importlib


def generate_id(length=8):
    """Generate a random valid Python filename."""
    letters = string.ascii_lowercase + string.digits
    return "".join(random.choices(letters, k=length))


def format_code(code: str) -> str:
    """Remove markdown code blocks if present."""
    if code.startswith("```python"):
        code = code.replace("```python", "", 1)
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def format_concepts(concepts: list[Concept]) -> str:
    return f"[{', '.join(f'Concept.{c.name}' for c in concepts)}]"


def format_description(description: str) -> str:
    formatted_lines = []
    current_line = ""
    for word in description.split():
        if len(current_line) + len(word) + 1 <= 100:
            current_line += " " + word if current_line else word
        else:
            formatted_lines.append(current_line)
            current_line = word
    if current_line:
        formatted_lines.append(current_line)
    return "\n".join(formatted_lines)


def create_puzzle(concepts: list[Concept], description: str, code: str):
    """Generate a puzzle file with random name containing the provided content.

    Args:
        concepts: List of Concept enums defining puzzle concepts
        description: String containing puzzle description
        code: String containing the puzzle code (may be wrapped in markdown)
    """
    # Generate random filename
    id = generate_id()
    print(f"generating {id}")

    # Clean up code string clean_code = format_code(code)

    content = f'''# Auto-generated puzzle file
from arc.core import Concept

id = "{id}"

concepts = {format_concepts(concepts)}

description = """{format_description(description)}"""

{format_code(code)}
    '''
    module_name = f"{'_'.join(concepts)}_{id}"
    fname = f"arc/tasks/generated_seed_tasks/{module_name}.py"
    # Write the file
    with open(fname, "w") as f:
        f.write(content)
    return importlib.import_module(f"arc.tasks.generated_seed_tasks.{module_name}")


def create_task(id, generate_input, solve, train_count=3):
    examples = []
    for _ in range(train_count + 1):
        input = generate_input()
        output = solve(input)
        examples.append(dict(input=input, output=output))
    return Task(id, train=examples[:-1], test=[examples[-1]])


def generate_puzzle(concepts: list[Concept]):
    description = openai.complete_structured(
        prompts.create_puzzle_descriptions_prompt(concepts, 1),
        prompts.PuzzleDescriptions,
        system_prompt=prompts.puzzlemaker_role_prompt,
        temperature=1,
    ).descriptions[0]

    code_output = openai.complete(
        system_prompt=prompts.programmer_role_prompt,
        prompt=prompts.create_puzzle_code_prompt(description),
    )

    puzzle = create_puzzle(concepts, description, code_output)
    return create_task(puzzle.id, puzzle.generate_input, puzzle.solve)
