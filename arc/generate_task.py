from pydantic.main import BaseModel


class Task(BaseModel):
    concepts: list[str]  # TODO should this be a string?
    description: list[str]
