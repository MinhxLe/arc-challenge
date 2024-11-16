from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern recognition, grid transformation

# description:
# In the input, you will see a grid with a repeated pattern of colored pixels arranged in a circular shape.
# The output should identify the pattern and replicate it in a square region of the same size centered in the output grid.
# The output grid will be larger than the input grid to accommodate the square pattern.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the connected components in the input grid
    objects = find_connected_components(input_grid, monochromatic=False, connectivity=8)
    
    # Assume there is only one pattern in the grid
    if not objects:
        return np.full(input_grid.shape, Color.BLACK)  # Return a black grid if no patterns are found

    # Get the first detected object
    pattern = objects[0]
    bounding_box_coords = bounding_box(pattern, background=Color.BLACK)
    x, y, width, height = bounding_box_coords

    # Crop the pattern from the input grid
    cropped_pattern = crop(pattern, background=Color.BLACK)

    # Determine the size of the output grid
    output_size = max(width, height) + 2  # Increase size to accommodate the pattern in the center
    output_grid = np.full((output_size, output_size), Color.BLACK)

    # Calculate the position to place the cropped pattern in the output grid
    start_x = (output_size - cropped_pattern.shape[0]) // 2
    start_y = (output_size - cropped_pattern.shape[1]) // 2

    # Place the cropped pattern in the output grid
    blit_sprite(output_grid, cropped_pattern, x=start_x, y=start_y, background=Color.BLACK)

    return output_grid