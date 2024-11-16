from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern generation, color mapping, grid manipulation

# description:
# In the input, you will see a grid with a specific pattern of colored pixels. 
# To create the output, you need to replicate this pattern in a larger grid while changing the color of the new pixels based on their distance from the center of the original pattern. 
# The closer a pixel is to the center, the more its color should be shifted towards a specified color palette.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Determine the dimensions of the input grid
    height, width = input_grid.shape

    # Create an output grid with the same dimensions as the input grid, filled with the background color
    output_grid = np.full((height, width), Color.BLACK)

    # Find the center of the original pattern
    center_x, center_y = height // 2, width // 2

    # Define a color palette that will be used for the new colors
    color_palette = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GRAY]

    # Get all coordinates of the original pattern (non-background pixels)
    pattern_coords = np.argwhere(input_grid!= Color.BLACK)

    # Calculate the maximum distance from the center for color mapping
    max_distance = max(np.linalg.norm([x - center_x, y - center_y]) for x, y in pattern_coords)

    # Iterate through each pixel in the output grid
    for x in range(height):
        for y in range(width):
            # Calculate the distance from the center of the original pattern
            distance = np.linalg.norm([x - center_x, y - center_y])
            # Normalize the distance to a value between 0 and 1
            normalized_distance = distance / max_distance

            # Map the normalized distance to a color index
            color_index = int(normalized_distance * len(color_palette))

            # If the current pixel is part of the original pattern, copy its color to the output grid
            if (x, y) in pattern_coords:
                output_grid[x, y] = input_grid[x, y]
            else:
                # Determine the color to use based on the distance
                output_grid[x, y] = color_palette[color_index % len(color_palette)]

    return output_grid