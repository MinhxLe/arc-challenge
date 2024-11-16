from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern recognition, filling, symmetry

# description:
# In the input you will see a grid with a repeating pattern of colored pixels. 
# To create the output, identify the repeating pattern and fill in any gaps in the pattern with a specified color, 
# effectively completing the pattern while maintaining its symmetry.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Get the unique colors in the grid, excluding the background
    unique_colors = np.unique(input_grid)
    unique_colors = unique_colors[unique_colors!= Color.BLACK]
    
    # Create a copy of the input grid to fill in the gaps
    output_grid = np.copy(input_grid)

    # Detect the connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4)

    # Fill in the gaps with the first unique color found
    for obj in objects:
        # Get the bounding box of the object
        x, y, w, h = bounding_box(obj, background=Color.BLACK)
        # Check if the bounding box area is empty (only background)
        if np.all(obj == Color.BLACK):
            # Fill the bounding box with the first unique color
            for i in range(h):
                for j in range(w):
                    if output_grid[y + i, x + j] == Color.BLACK:
                        output_grid[y + i, x + j] = unique_colors[0]  # Fill with the first unique color

    return output_grid