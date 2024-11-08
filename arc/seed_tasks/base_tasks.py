from arc.seed_tasks.types import SeedTask, Concept, Grid
from arc.utils import ndarray_to_tuple

import numpy as np

rng = np.random.default_rng()


def hmirror_grid_transform(input_grid: Grid) -> Grid:
    """
    Horizontally mirrors a grid by reversing the order of rows while maintaining column order.

    Args:
        input_grid: A tuple of tuples representing a 2D grid where each inner tuple is a row

    Returns:
        A new grid with rows in reversed order (horizontally mirrored)
    """
    # Convert the input grid to a list of rows for clarity
    rows = list(input_grid)

    # Create a new list with rows in reversed order
    # This mirrors the grid horizontally (top becomes bottom, bottom becomes top)
    mirrored_rows = rows[::-1]

    # Convert back to tuple of tuples to maintain Grid type
    return tuple(mirrored_rows)


def hmirror_grid_generate() -> Grid:
    # Randomly sample the height and width of the grid
    # from between the min and max size.
    height = rng.integers(low=1, high=30, endpoint=True)
    width = rng.integers(low=1, high=30, endpoint=True)

    # Randomly assign integer values to each element of a grid
    # sized using the random height and width.
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
