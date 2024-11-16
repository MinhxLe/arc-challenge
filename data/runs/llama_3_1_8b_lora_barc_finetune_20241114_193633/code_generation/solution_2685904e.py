from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel shifting, color replacement, wrapping

# description:
# In the input, you will see a grid filled with colored pixels, where one color represents a "teal" pixel that will be the anchor.
# The task is to shift all colored pixels away from the teal pixel in all four cardinal directions (up, down, left, right) until they hit another color or the edge of the grid.
# Additionally, replace the teal pixel with a new color of your choice.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)
    
    # Find the position of the teal pixel
    teal_pixel_coords = np.argwhere(input_grid == Color.BLUE)  # Assuming blue is the teal color
    teal_pixel = teal_pixel_coords[0][0] if teal_pixel_coords.size > 0 else None
    if teal_pixel is None:
        return output_grid  # No teal pixel found, return original grid

    teal_x, teal_y = teal_pixel

    # Define the new color to replace the teal pixel
    new_color = Color.GREEN  # This can be any color of your choice

    # Replace the teal pixel with the new color
    output_grid[teal_x, teal_y] = new_color

    # Get the coordinates of the colored pixels
    colored_pixels = np.argwhere(input_grid!= Color.BLACK)
    
    # Define the direction vectors for shifting
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    # Shift each colored pixel away from the teal pixel
    for dx, dy in directions:
        for x, y in colored_pixels:
            if (0 <= x + dx < output_grid.shape[0]) and (0 <= y + dy < output_grid.shape[1]):
                if output_grid[x + dx, y + dy] == Color.BLACK:  # If there's no pixel in the direction
                    output_grid[x + dx, y + dy] = output_grid[x, y]  # Move pixel
                    output_grid[x, y] = Color.BLACK  # Clear the original position

    return output_grid