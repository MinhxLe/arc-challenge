from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel extraction, color mapping, grid transformation

# description:
# In the input, you will see a grid filled with colored pixels, including a single distinct color that forms a shape or pattern. 
# The shape can be of any form but must occupy a connected region. 
# To make the output, you should extract the shape from the input grid and color it based on its color mapping:
# - The color of the shape will be mapped to a new color based on its position:
#   - If the shape is in the upper half of the grid, change its color to blue.
#   - If the shape is in the lower half of the grid, change its color to yellow.
# The output grid should maintain the same dimensions as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Detect the shape in the input grid
    shape_objects = detect_objects(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # Initialize the output grid with the same size as the input
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # For each detected shape, color it based on its position
    for obj in shape_objects:
        # Crop the shape to isolate it
        shape = crop(obj, background=Color.BLACK)

        # Get the bounding box of the shape
        x, y, width, height = bounding_box(shape)

        # Determine the new color based on the position of the shape in the grid
        if y < input_grid.shape[0] // 2:
            new_color = Color.BLUE
        else:
            new_color = Color.YELLOW

        # Color the shape in the output grid
        for i in range(shape.shape[0]):
            for j in range(shape.shape[1]):
                if shape[i, j]!= Color.BLACK:
                    output_grid[x + i, y + j] = new_color

    return output_grid