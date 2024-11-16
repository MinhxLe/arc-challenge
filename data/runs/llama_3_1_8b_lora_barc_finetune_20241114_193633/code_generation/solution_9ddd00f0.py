from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, pixel filling, grid manipulation

# description:
# In the input, you will see a grid with a symmetric shape made up of colored pixels.
# To create the output, fill the area outside the shape with the same color as the shape's color,
# ensuring that the filling respects the symmetry of the shape.

def transform(input_grid):
    # Detect the color of the shape in the grid
    shape_color = np.unique(input_grid[input_grid!= Color.BLACK])[0]

    # Find the bounding box of the shape
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)
    
    # Create the output grid with the same size as input
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Fill the output grid with the shape's color
    output_grid[x:x + width, y:y + height] = shape_color

    # Fill the symmetric area outside the shape with the same color
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):
            if output_grid[i, j] == Color.BLACK:
                # Check if the pixel is symmetric with respect to the center of the shape
                if (i < height // 2 and j < width // 2) or (i >= height // 2 and j >= width // 2):
                    output_grid[i, j] = shape_color

    return output_grid