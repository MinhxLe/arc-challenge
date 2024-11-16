from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, distance mapping, boundary detection

# description:
# In the input, you will see a grid with a black background and colored pixels scattered throughout.
# To make the output, for each colored pixel, create a boundary around it using a new color, 
# and fill the boundary with a gradient of colors based on the distance from the original pixel.
# The closer the pixel is to the original, the closer the color should be to the original pixel's color.

def transform(input_grid):
    # Get the dimensions of the grid
    height, width = input_grid.shape
    output_grid = np.copy(input_grid)

    # Find the coordinates of all colored pixels
    colored_pixels = np.argwhere(input_grid!= Color.BLACK)

    # For each colored pixel, create a boundary
    for x, y in colored_pixels:
        # Get the color of the current pixel
        color = input_grid[x, y]

        # Draw a boundary around the pixel
        for dx in range(-1, 2):  # -1, 0, 1
            for dy in range(-1, 2):  # -1, 0, 1
                if dx == 0 and dy == 0:
                    continue  # Skip the center pixel
                if 0 <= x + dx < height and 0 <= y + dy < width:
                    # Calculate the distance from the current pixel
                    distance = max(abs(dx), abs(dy))  # Manhattan distance
                    # Determine the new color based on distance
                    boundary_color = color if distance == 0 else Color.GRAY  # Default to gray for farther distances

                    # Fill the boundary pixel
                    output_grid[x + dx, y + dy] = boundary_color

    return output_grid