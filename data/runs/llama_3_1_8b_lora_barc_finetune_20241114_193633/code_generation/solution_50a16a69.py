from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color extraction, pattern replication

# description:
# In the input, you will see a grid containing a colorful pattern in the center and a colored border around it.
# To create the output grid, you should extract the central colorful pattern, replicate it in a grid that matches the size of the border,
# and fill the border with the color of the central pattern.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # First, we find the bounding box of the colorful pattern
    pattern = crop(input_grid, background=Color.BLACK)

    # Create an output grid of the same size as the border
    output_grid = np.full((input_grid.shape[0] - 2, input_grid.shape[1] - 2), Color.BLACK)

    # Fill the border with the color of the central pattern
    border_color = pattern[0, 0]  # Assuming the border and central pattern share the same color

    # Place the central pattern into the output grid
    output_grid = blit_sprite(output_grid, pattern, x=1, y=1)

    # Fill the border with the color of the central pattern
    output_grid[0, :] = border_color  # Top border
    output_grid[-1, :] = border_color  # Bottom border
    output_grid[:, 0] = border_color  # Left border
    output_grid[:, -1] = border_color  # Right border

    return output_grid