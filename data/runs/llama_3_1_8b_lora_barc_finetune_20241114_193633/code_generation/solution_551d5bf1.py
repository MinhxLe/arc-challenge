from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry, object transformation, color mapping

# description:
# In the input, you will see a colored shape that is symmetric along the vertical axis.
# To make the output, reflect the shape across the vertical axis and color the new pixels with a different color.
# The color of the new pixels should be determined by the color of the original shape at the corresponding positions.

def transform(input_grid):
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Get the dimensions of the grid
    height, width = input_grid.shape

    # Find the color of the original shape (non-background color)
    original_color = np.unique(input_grid)[input_grid!= Color.BLACK][0]

    # Iterate over each pixel in the input grid
    for x in range(height):
        for y in range(width):
            # Check if the current pixel is not the background
            if input_grid[x, y]!= Color.BLACK:
                # Calculate the reflected position across the vertical axis
                reflected_x = x
                reflected_y = width - 1 - y

                # Set the reflected position in the output grid with the original color
                output_grid[reflected_x, reflected_y] = original_color

    return output_grid