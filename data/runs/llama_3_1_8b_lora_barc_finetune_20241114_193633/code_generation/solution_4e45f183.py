from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern extraction, color replacement

# description:
# In the input, you will see a grid with a complex pattern made of multiple colors, and a small colored square object at the center.
# To create the output, extract the pattern around the square object and replace its color with the color of the square.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to black
    output_grid = np.zeros_like(input_grid)

    # Find the center of the grid
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2

    # Get the color of the square object at the center
    square_color = input_grid[center_x, center_y]

    # Create a mask for the square object
    square_mask = (input_grid == square_color)

    # Extract the surrounding pattern
    pattern = np.copy(input_grid)

    # Replace the color of the square object with the background color (black)
    pattern[square_mask] = Color.BLACK

    # Crop the pattern to get the surrounding pattern
    surrounding_pattern = crop(pattern, background=Color.BLACK)

    # Replace the color of the square object in the output grid
    output_grid[square_mask] = square_color

    # Place the surrounding pattern back into the output grid
    output_grid = blit_sprite(output_grid, surrounding_pattern, x=center_x - surrounding_pattern.shape[0] // 2, y=center_y - surrounding_pattern.shape[1] // 2, background=Color.BLACK)

    return output_grid