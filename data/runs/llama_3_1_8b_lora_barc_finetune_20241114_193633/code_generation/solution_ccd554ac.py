from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# tiling, repetition, symmetry

# description:
# In the input, you will see a small colored tile pattern. To make the output, replicate the tile pattern across the entire grid, ensuring that the tiles align perfectly without any gaps or overlaps. The output grid will be larger than the input grid to accommodate the repeated tiles.

def transform(input_grid):
    # Get the dimensions of the input grid
    input_height, input_width = input_grid.shape

    # Determine the size of the output grid, which is a multiple of the input grid size
    output_height = input_height * 3  # Repeat the height 3 times
    output_width = input_width * 3    # Repeat the width 3 times
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Fill the output grid with repeated tiles
    for i in range(3):  # Repeat vertically 3 times
        for j in range(3):  # Repeat horizontally 3 times
            blit_sprite(output_grid, input_grid, x=j * input_width, y=i * input_height, background=Color.BLACK)

    return output_grid