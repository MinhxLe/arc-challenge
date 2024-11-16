from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color extraction, pattern detection, color mapping

# description:
# In the input, you will see a grid containing a repetitive pattern of colored pixels. 
# To make the output, extract all unique colors from the input grid and create a new grid that 
# displays these colors in a single row, maintaining the order of appearance from the input.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Extract unique colors from the input grid
    unique_colors = []
    seen = set()

    for row in input_grid:
        for color in row:
            if color not in seen and color!= Color.BLACK:
                seen.add(color)
                unique_colors.append(color)

    # Create the output grid with the same number of columns as unique colors
    output_grid = np.zeros((1, len(unique_colors)), dtype=int)

    # Fill the output grid with the unique colors
    for i, color in enumerate(unique_colors):
        output_grid[0, i] = color

    return output_grid