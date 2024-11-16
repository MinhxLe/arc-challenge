from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, symmetry detection

# description:
# In the input, you will see a grid with a symmetric pattern made of two colors. 
# The pattern is symmetric across both axes. To create the output, swap the two colors of the pattern, 
# and then rotate the entire grid by 90 degrees clockwise. 
# The output grid will have the same dimensions as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Identify the two colors used in the input grid
    unique_colors = np.unique(input_grid)
    color1, color2 = unique_colors[1], unique_colors[0]  # Assuming there are exactly two colors excluding background

    # Swap the two colors in the output grid
    output_grid[output_grid == color1] = Color.BLACK
    output_grid[output_grid == color2] = color1
    output_grid[output_grid == Color.BLACK] = color2

    # Rotate the output grid by 90 degrees clockwise
    output_grid = np.rot90(output_grid, k=-1)

    return output_grid