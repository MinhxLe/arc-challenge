from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern extraction, tiling, grid transformation

# description:
# In the input, you will see a grid with a specific pattern repeated multiple times. 
# The pattern is a 2x2 grid of colored pixels, and the output grid should be a larger grid where each
# instance of the original pattern is replicated in a larger grid, maintaining the relative positions of the
# original pattern. The output grid should be large enough to fit all instances of the pattern.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # Assume there's only one pattern in the input for simplicity
    assert len(objects) == 1
    pattern = objects[0]

    # Get the bounding box of the pattern
    pattern_box = bounding_box(pattern, background=Color.BLACK)
    pattern_height, pattern_width = pattern_box[3], pattern_box[2]

    # Calculate the size of the output grid
    output_height = (input_grid.shape[0] // pattern_height) * pattern_height
    output_width = (input_grid.shape[1] // pattern_width) * pattern_width
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Blit the pattern into the output grid at each position where it appears
    for x in range(0, output_height, pattern_height):
        for y in range(0, output_width, pattern_width):
            # Crop the pattern from the input grid
            sprite = crop(pattern, background=Color.BLACK)
            blit_sprite(output_grid, sprite, x=x, y=y, background=Color.BLACK)

    return output_grid