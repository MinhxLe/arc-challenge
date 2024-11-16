from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern extraction, color mapping, symmetry

# description:
# In the input, you will see a grid with a central pattern surrounded by a border of colored pixels.
# The border consists of pixels of different colors. The output should be a grid where the central pattern is
# replicated in all four quadrants of the output grid, and the color of the central pattern should be replaced
# with the corresponding color from the border. The colors from the border should be mapped to the central pattern
# in a clockwise order starting from the top-left corner of the border.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Extract the central pattern and the border colors
    output_grid = np.copy(input_grid)

    # Crop the grid to get the central pattern
    central_pattern = crop(output_grid, background=Color.BLACK)

    # Get the border colors in clockwise order
    top_left = output_grid[0, 0]
    top_right = output_grid[0, -1]
    bottom_right = output_grid[-1, -1]
    bottom_left = output_grid[-1, 0]

    # Create a color mapping for the central pattern
    color_mapping = {
        Color.RED: top_left,
        Color.BLUE: top_right,
        Color.GREEN: bottom_right,
        Color.YELLOW: bottom_left,
    }

    # Determine the size of the output grid
    output_height = central_pattern.shape[0]
    output_width = central_pattern.shape[1]
    output_grid = np.full((output_height * 2, output_width * 2), Color.BLACK)

    # Place the central pattern in all four quadrants
    for i in range(2):
        for j in range(2):
            # Get the color to use for the quadrant
            color = color_mapping.get(central_pattern[0, 0], Color.BLACK)
            # Replace the color in the central pattern
            colored_pattern = np.where(central_pattern!= Color.BLACK, color, Color.BLACK)
            # Blit the colored pattern to the output grid
            blit_sprite(output_grid, colored_pattern, x=i * output_height, y=j * output_width, background=Color.BLACK)

    return output_grid