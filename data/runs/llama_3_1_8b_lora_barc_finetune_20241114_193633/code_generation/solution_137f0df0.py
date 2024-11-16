from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern recognition, filling, symmetry

# description:
# In the input, you will see a grid containing a symmetrical pattern of colored pixels on a black background.
# The output should replicate this pattern in all four quadrants of the grid while maintaining the original colors.
# The output grid should be larger than the input grid to accommodate the replication.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the bounding box of the pattern in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4)
    
    # Assume there's only one object that we want to replicate
    pattern = objects[0]
    x, y, width, height = bounding_box(pattern, background=Color.BLACK)
    
    # Create the output grid with size double the input grid size
    output_grid = np.full((height * 2, width * 2), Color.BLACK)

    # Crop the pattern to remove any extra black pixels
    sprite = crop(pattern, background=Color.BLACK)

    # Place the original pattern in the center of the output grid
    blit_sprite(output_grid, sprite, x=width // 2, y=height // 2, background=Color.BLACK)

    # Copy the pattern to the other quadrants
    blit_sprite(output_grid, sprite, x=0, y=height // 2, background=Color.BLACK)  # Top-left
    blit_sprite(output_grid, sprite, x=width, y=height // 2, background=Color.BLACK)  # Top-right
    blit_sprite(output_grid, sprite, x=width // 2, y=0, background=Color.BLACK)  # Bottom-left
    blit_sprite(output_grid, sprite, x=width, y=0, background=Color.BLACK)  # Bottom-right

    # Color the border of the quadrants with a different color
    border_color = Color.BLUE
    output_grid[0, :] = border_color  # Top border
    output_grid[-1, :] = border_color  # Bottom border
    output_grid[:, 0] = border_color  # Left border
    output_grid[:, -1] = border_color  # Right border

    return output_grid