from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern extraction, color blending, grid manipulation

# description:
# In the input, you will see a grid containing a distinct pattern made up of colored pixels surrounded by a black background. 
# The output grid should extract the distinct pattern and blend its colors with the black background. 
# The blended color at each pixel will be determined by averaging the colors of the surrounding pixels (up, down, left, right) 
# that are not black. If a pixel has no surrounding non-black pixels, it will remain black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to black
    output_grid = np.zeros_like(input_grid)

    # Get the dimensions of the input grid
    rows, cols = input_grid.shape

    # Iterate over each pixel in the grid
    for x in range(rows):
        for y in range(cols):
            # Check if the current pixel is not black
            if input_grid[x, y]!= Color.BLACK:
                # Get the surrounding pixels
                surrounding_colors = []
                if x > 0:  # up
                    surrounding_colors.append(input_grid[x-1, y])
                if x < rows - 1:  # down
                    surrounding_colors.append(input_grid[x+1, y])
                if y > 0:  # left
                    surrounding_colors.append(input_grid[x, y-1])
                if y < cols - 1:  # right
                    surrounding_colors.append(input_grid[x, y+1])

                # Filter out the black pixels
                surrounding_colors = [color for color in surrounding_colors if color!= Color.BLACK]

                # If there are surrounding colors, blend them
                if surrounding_colors:
                    # Average the colors (assuming they can be represented as integers)
                    avg_color = sum(surrounding_colors) // len(surrounding_colors)
                    output_grid[x, y] = avg_color
                else:
                    output_grid[x, y] = Color.BLACK

    return output_grid