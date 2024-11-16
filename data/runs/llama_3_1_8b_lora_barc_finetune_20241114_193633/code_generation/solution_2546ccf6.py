from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, object detection, symmetry

# description:
# In the input, you will see a grid containing various colored objects and a black background.
# To create the output, detect all colored objects in the grid and replace their colors with a new color that is the average of their original colors.
# The output grid should maintain the same structure as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Initialize the output grid as a copy of the input grid
    output_grid = np.copy(input_grid)

    # Detect all colored objects in the grid (excluding black)
    objects = detect_objects(grid=input_grid, background=Color.BLACK, monochromatic=True, connectivity=4)

    # For each detected object, calculate the average color
    for obj in objects:
        # Find the unique colors in the object
        unique_colors = np.unique(obj[obj!= Color.BLACK])
        
        if len(unique_colors) > 0:
            # Calculate the average color (this could be done using a weighted average if needed)
            average_color = np.mean(unique_colors).astype(int)

            # Replace the colors of the object with the average color in the output grid
            output_grid[obj!= Color.BLACK] = average_color

    return output_grid