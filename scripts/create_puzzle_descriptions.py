import random
from arc.tasks import prompts
from arc.core import Concept
from arc.external import openai

concepts = random.choices([c.value for c in Concept], k=2)

output = openai.complete_structured(
    prompts.create_puzzle_descriptions_prompt(concepts, 5), prompts.PuzzleDescriptions
)
