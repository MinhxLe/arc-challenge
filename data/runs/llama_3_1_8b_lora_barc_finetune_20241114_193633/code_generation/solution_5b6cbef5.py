from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern replication, color transformation

# description:
# In the input, you will see a small pattern of colored pixels in the center of the grid.
# To create the output, replicate this pattern to fill a larger grid, maintaining the original color arrangement,
# while ensuring that the edges of the replicated patterns align perfectly.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Get the central pattern from the input grid
    pattern = crop(input_grid, background=Color.BLACK)
    
    # Determine the size of the output grid
    output_size = (input_grid.shape[0] * 3, input_grid.shape[1] * 3)  # 3x replication
    output_grid = np.full(output_size, Color.BLACK)

    # Calculate the starting position for the pattern
    for i in range(3):
        for j in range(3):
            # Calculate the position to place the pattern
            x_offset = i * input_grid.shape[0]
            y_offset = j * input_grid.shape[1]
            blit_sprite(output_grid, pattern, x=x_offset, y=y_offset, background=Color.BLACK)

    return output_grid