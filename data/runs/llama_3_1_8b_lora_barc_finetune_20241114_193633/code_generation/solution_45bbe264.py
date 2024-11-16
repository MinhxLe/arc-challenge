from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# radial symmetry, color mapping, color gradient

# description:
# In the input, you will see a grid with a central colored pixel and several colored pixels around it. 
# To create the output, map the colors of the surrounding pixels to the colors of the pixels in a circular pattern around the central pixel,
# creating a radial symmetry effect.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid of the same size as the input
    output_grid = np.copy(input_grid)
    
    # Find the central pixel (the first non-black pixel)
    central_pixel = np.argwhere(input_grid!= Color.BLACK)
    if len(central_pixel) == 0:
        return output_grid  # No central pixel found, return original grid
    
    center_x, center_y = central_pixel[0]  # Get the coordinates of the central pixel

    # Get surrounding pixels (in a circular pattern)
    surrounding_positions = [
        (center_x - 1, center_y), (center_x + 1, center_y),
        (center_x, center_y - 1), (center_x, center_y + 1),
        (center_x - 1, center_y - 1), (center_x + 1, center_y + 1),
        (center_x - 1, center_y + 1), (center_x + 1, center_y - 1),
        (center_x, center_y)  # Center itself
    ]

    # Map surrounding colors to the central pixel
    for (x, y) in surrounding_positions:
        if 0 <= x < input_grid.shape[0] and 0 <= y < input_grid.shape[1]:
            surrounding_color = input_grid[x, y]
            output_grid[x, y] = surrounding_color

    # Apply radial symmetry
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue  # Skip the center pixel
            # Calculate the symmetric position
            sym_x = center_x + dx
            sym_y = center_y + dy
            
            # Assign the color from the surrounding pixel to its symmetric position
            if 0 <= sym_x < output_grid.shape[0] and 0 <= sym_y < output_grid.shape[1]:
                output_grid[sym_x, sym_y] = output_grid[center_x, center_y]

    return output_grid