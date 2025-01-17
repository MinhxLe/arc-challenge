from arc.core import (
    MIN_GRID_WIDTH,
    MIN_GRID_HEIGHT,
    MAX_GRID_WIDTH,
    MAX_GRID_HEIGHT,
    Grid,
    Color,
    Concept,
)
import numpy as np

concepts = [Concept.REFLECTION, Concept.SYMMETRY]

description = """The input is a grid and the output is a grid.
To make the output grid, create the mirror image of the input grid using
the horizontal midline as the axis of reflection. If the height of the input grid
is odd, the middle row remains unchanged. In essence, simply reverse
the ordering of the rows."""


def transform_grid(input_grid: Grid) -> Grid:
    """
    Horizontally mirrors a grid by reversing the order of rows while maintaining column order.

    Args:
        input_grid: A tuple of tuples representing a 2D grid where each inner tuple is a row

    Returns:
        A new grid with rows in reversed order (horizontally mirrored)
    """

    return input_grid[::-1]


def generate_input() -> Grid:
    rng = np.random.default_rng()

    # Randomly sample the height and width of the grid
    # from between the min and max size.
    height = rng.integers(low=MIN_GRID_HEIGHT, high=MAX_GRID_HEIGHT + 1)
    width = rng.integers(low=MIN_GRID_WIDTH, high=MAX_GRID_WIDTH + 1)

    # Randomly assign integer values to each element of a grid
    # sized using the random height and width.
    return rng.integers(
        low=min(Color),
        high=max(Color) + 1,
        size=(height, width),
    )
