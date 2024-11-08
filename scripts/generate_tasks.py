from pydantic.main import BaseModel
from arc.external import openai


class Task(BaseModel):
    concepts: list[str]  # TODO should this be a string?
    description: str


class TaskList(BaseModel):
    tasks: list[Task]


t1 = Task(concepts=["symmetry"], description="Reflects a grid horizontally.")
t2 = Task(concepts=["color"], description="Makes all objects blue.")
t3 = Task(
    concepts=["color", "symmetry"],
    description="Makes all objects blue and reflect vertical axis.",
)

tasks = [t1, t2]
task_string = "\n".join([t.model_dump_json() for t in tasks])

# TODO add constraints on concepts, grid size, color
system_prompt = """
You are a puzzle maker designing geometric, physical, and topological
puzzles for curious middle-schoolers.

Each puzzle consists of uncovering a deterministic rule, pattern, procedure,
algorithm, or transformation law that maps inputs to outputs.
Both the inputs and outputs are 2D grids of colored pixels. There are 10
colors, but the order of the colors is never relevant to the puzzle. 
"""

prompt = f"""
You generated these tasks on previous requests.

{task_string}

Brainstorm 3 more, using similar thinking, outputing a list of json.
"""

output = openai.complete_structured(
    prompt,
    TaskList,
    system_msg=system_prompt,
)
