from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# layering, transparency

# description:
# In the input, you will see multiple layers of colored pixels stacked on top of each other. 
# To make the output, create a new grid where each layer is represented as a transparent overlay of the original colors, 
# and the resulting output grid will show the combined effect of all layers.

def transform(input_grid):
    # Initialize the output grid with the background color
    output_grid = np.zeros_like(input_grid)

    # Iterate over each pixel in the input grid
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            # If the pixel is not the background, we will overlay it onto the output grid
            if input_grid[x, y]!= Color.BLACK:
                output_grid[x, y] = input_grid[x, y]  # Keep the color

    # Now we need to blend the colors in the output grid
    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            if output_grid[x, y]!= Color.BLACK:
                # Check if there are any other colors in the same position
                if np.any(output_grid == Color.BLACK):
                    # If there are no other colors, just keep the current color
                    continue
                # If there are other colors, we can average the colors (simple blending)
                color_counts = {}
                for i in range(output_grid.shape[0]):
                    for j in range(output_grid.shape[1]):
                        if (i!= x or j!= y) and output_grid[i, j]!= Color.BLACK:
                            color = output_grid[i, j]
                            if color in color_counts:
                                color_counts[color] += 1
                            else:
                                color_counts[color] = 1
                # Select the most frequent color as the blended color
                blended_color = max(color_counts, key=color_counts.get)
                output_grid[x, y] = blended_color

    return output_grid