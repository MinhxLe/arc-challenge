import typing as ta
from enum import Enum


Grid = ta.Tuple[ta.Tuple[int]]


class Concept(str, Enum):
    COLOR_MAPPING = "color mapping"
    REFLECTION = "reflection"
    ROTATION = "rotation"
    SYMMETRY = "symmetry"
    TRANSLATION = "translation"


class SeedTask:
    def __init__(
        self,
        concepts: ta.List[Concept],
        description: str,
        transform_grid: ta.Callable[..., Grid],
        generate_input: ta.Callable[[], Grid],
    ):
        self.concepts = concepts
        self.description = description
        self.transform_grid = transform_grid
        self.generate_input = generate_input
