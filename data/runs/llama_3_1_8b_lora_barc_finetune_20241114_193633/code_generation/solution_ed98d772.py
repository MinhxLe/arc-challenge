from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# symmetry, mirroring

# description:
# In the input, you will see a grid with a central colored shape and a background of black pixels.
# To create the output, mirror the shape across both the horizontal and vertical axes,
# filling in the mirrored regions with the same color as the original shape.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the bounding box of the colored shape
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)

    # Extract the shape from the input grid
    shape = input_grid[y:y + height, x:x + width]

    # Create the output grid, which is twice the size of the shape
    output_grid = np.full((height * 2, width * 2), Color.BLACK)

    # Fill the output grid with the original shape and its mirrored versions
    blit_sprite(output_grid, shape, x=0, y=0, background=Color.BLACK)  # Top-left
    blit_sprite(output_grid, shape, x=width, y=0, background=Color.BLACK)  # Top-right
    blit_sprite(output_grid, shape, x=0, y=height, background=Color.BLACK)  # Bottom-left
    blit_sprite(output_grid, shape, x=width, y=height, background=Color.BLACK)  # Bottom-right

    # Fill the mirrored regions
    for dx in [width, 0]:  # 0 for horizontal, width for vertical
        for dy in [height, 0]:  # 0 for vertical, height for horizontal
            mirrored_shape = shape[::-dy, ::-dx]  # Mirror both axes
            blit_sprite(output_grid, mirrored_shape, x=dx, y=dy, background=Color.BLACK)

    return output_grid