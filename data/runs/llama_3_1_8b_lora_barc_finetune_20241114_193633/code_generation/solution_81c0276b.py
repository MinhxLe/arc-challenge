from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern extraction, counting, color filling

# description:
# In the input, you will see a grid with various colored pixels arranged in a checkerboard pattern.
# To create the output, extract the unique colors present in the grid and fill the output grid with those colors in a new pattern,
# ensuring that the output grid is a smaller grid that matches the unique colors in the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Get unique colors in the input grid, excluding the background (black)
    unique_colors = np.unique(input_grid)
    unique_colors = unique_colors[unique_colors!= Color.BLACK]

    # Determine the size of the output grid
    output_size = len(unique_colors)
    output_grid = np.zeros((output_size, output_size), dtype=int)

    # Fill the output grid with the unique colors in a checkerboard pattern
    for i in range(output_size):
        for j in range(output_size):
            if (i + j) % 2 == 0:  # Fill only the even positions
                output_grid[i, j] = unique_colors[i % len(unique_colors)]

    return output_grid