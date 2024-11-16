from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color inversion, grid transformation

# description:
# In the input grid, you will see a pattern made of two colors: one for the shape and one for the background.
# To create the output grid, invert the colors of the shape while keeping the background color unchanged.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Get the unique colors in the grid, excluding the background
    unique_colors = np.unique(input_grid)
    background_color = Color.BLACK  # Assuming black is the background color

    # Ensure that there are exactly two colors (shape and background)
    assert len(unique_colors) == 3, "There should be exactly two colors besides the background."
    
    shape_color = unique_colors[0] if unique_colors[0]!= background_color else unique_colors[1]

    # Invert the shape color to the other unique color
    output_grid[output_grid == shape_color] = unique_colors[0] if unique_colors[0]!= background_color else unique_colors[1]

    return output_grid