from arc.seed_tasks.types import SeedTask, Concept, Grid
from arc.utils import ndarray_to_tuple

import numpy as np

rng = np.random.default_rng()


def hmirror_grid_transform(input_grid: Grid) -> Grid:
    return input_grid[::-1]  # type: ignore


def hmirror_grid_generate() -> Grid:
    height = rng.integers(low=1, high=30, endpoint=True)
    width = rng.integers(low=1, high=30, endpoint=True)
    return ndarray_to_tuple(
        rng.integers(low=0, high=9, endpoint=True, size=(height, width))
    )


hmirror_grid = SeedTask(
    concepts=[Concept.REFLECTION, Concept.SYMMETRY],
    description="""The input is a grid and the output is a grid.
To make the output grid, create the mirror image of the input grid using
the horizontal midpoint as the axis of reflection. If the height of the input grid
is odd, the middle row remains unchanged. In essence, simply reverse
the ordering of the rows.""",
    transform_grid=hmirror_grid_transform,
    generate_input=hmirror_grid_generate,
)
