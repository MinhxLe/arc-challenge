from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# reflection, symmetry, color matching

# description:
# In the input, you will see a grid with a colored shape and a few isolated pixels of different colors.
# To make the output, reflect the colored shape across the vertical center of the grid and replace the reflected pixels with the color of the original shape.

def transform(input_grid):
    # Get the dimensions of the grid
    height, width = input_grid.shape
    
    # Create an output grid initialized with the background color
    output_grid = np.full((height, width), Color.BLACK)

    # Find the bounding box of the colored shape
    x, y, w, h = bounding_box(input_grid!= Color.BLACK)

    # Extract the shape
    shape = input_grid[x:x + w, y:y + h]

    # Reflect the shape vertically
    reflected_shape = shape[:, ::-1]

    # Get the color of the original shape (assuming monochromatic)
    original_color = shape[shape!= Color.BLACK][0]

    # Blit the original shape into the output grid
    blit_sprite(output_grid, shape, x, y, background=Color.BLACK)
    
    # Blit the reflected shape into the output grid
    blit_sprite(output_grid, reflected_shape, x, y + w, background=Color.BLACK)

    # Replace the reflected pixels with the original color
    for i in range(reflected_shape.shape[0]):
        for j in range(reflected_shape.shape[1]):
            if reflected_shape[i, j]!= Color.BLACK:
                output_grid[i, y + j + (width - w - j)] = original_color

    return output_grid