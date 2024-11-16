from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel merging, color blending, averaging

# description:
# In the input, you will see a grid of colored pixels with some pixels being black (transparent). 
# The goal is to merge adjacent pixels of the same color by averaging their color values and replacing the black pixels with the averaged color.
# The output grid should maintain the same shape as the input grid, with merged colors replacing the black pixels.

def transform(input_grid):
    # Initialize the output grid with the same shape as the input grid
    output_grid = np.copy(input_grid)

    # Get the shape of the input grid
    rows, cols = input_grid.shape

    # Iterate through the grid, merging adjacent pixels
    for x in range(rows):
        for y in range(cols):
            # Check if the current pixel is black (transparent)
            if input_grid[x, y] == Color.BLACK:
                # Gather the colors of neighboring pixels (up, down, left, right)
                neighbor_colors = []
                if x > 0 and input_grid[x - 1, y]!= Color.BLACK:
                    neighbor_colors.append(input_grid[x - 1, y])  # up
                if x < rows - 1 and input_grid[x + 1, y]!= Color.BLACK:
                    neighbor_colors.append(input_grid[x + 1, y])  # down
                if y > 0 and input_grid[x, y - 1]!= Color.BLACK:
                    neighbor_colors.append(input_grid[x, y - 1])  # left
                if y < cols - 1 and input_grid[x, y + 1]!= Color.BLACK:
                    neighbor_colors.append(input_grid[x, y + 1])  # right

                # Average the colors of the neighboring pixels (if any)
                if neighbor_colors:
                    # Convert colors to RGB values for averaging
                    color_values = [np.array(c) for c in neighbor_colors]
                    avg_color = np.mean(color_values, axis=0).astype(int)
                    # Set the averaged color in the output grid
                    output_grid[x, y] = Color.RED if np.array_equal(avg_color, np.array([255, 0, 0])) else Color.BLUE
                else:
                    # If no neighbors, keep it black
                    output_grid[x, y] = Color.BLACK

    return output_grid