from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern extraction, color mapping

# description:
# In the input, you will see a grid with a pattern of colored pixels arranged in a 2x2 formation in the center of the grid, surrounded by a border of black pixels. 
# To make the output, you should extract the 2x2 pattern from the center of the input grid and map its colors to a new color palette. 
# The new colors should be chosen from a predefined palette, and the output grid should have the same size as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Crop the central 2x2 pattern from the input grid
    center_x = input_grid.shape[0] // 2
    center_y = input_grid.shape[1] // 2
    pattern = input_grid[center_x - 1:center_x + 1, center_y - 1:center_y + 1]

    # Define a color mapping
    color_mapping = {
        Color.BLUE: Color.RED,
        Color.RED: Color.GREEN,
        Color.GREEN: Color.YELLOW,
        Color.YELLOW: Color.BLUE,
        Color.PINK: Color.BROWN,
        Color.BROWN: Color.PURPLE,
        Color.PURPLE: Color.ORANGE,
        Color.ORANGE: Color.GRAY,
        Color.GRAY: Color.BLACK,
        Color.BLACK: Color.BLACK,
    }

    # Create the output grid with the same size as the input
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Apply the color mapping to the pattern
    for i in range(2):
        for j in range(2):
            original_color = pattern[i, j]
            new_color = color_mapping.get(original_color, original_color)
            output_grid[center_x - 1 + i, center_y - 1 + j] = new_color

    return output_grid