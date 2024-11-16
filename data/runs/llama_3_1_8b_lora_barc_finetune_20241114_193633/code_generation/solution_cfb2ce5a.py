from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling

# description:
# In the input, you will see a grid with various colored shapes and a single black pixel located at (x, y).
# To create the output, fill the area surrounding the black pixel with the colors of the shapes that are symmetric to it.
# The output grid should be the same size as the input grid.

def transform(input_grid):
    # Make a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Find the position of the black pixel
    black_pixel_position = np.argwhere(input_grid == Color.BLACK)

    if black_pixel_position.size == 0:
        return output_grid  # No black pixel found, return original grid

    x, y = black_pixel_position[0]

    # Detect mirror symmetries in the grid
    symmetries = detect_mirror_symmetry(input_grid, ignore_colors=[Color.BLACK])

    # Fill the area surrounding the black pixel with the colors of the shapes that are symmetric to it
    for x_offset in range(-1, 2):  # Check surrounding pixels
        for y_offset in range(-1, 2):
            if x + x_offset >= 0 and x + x_offset < input_grid.shape[0] and y + y_offset >= 0 and y + y_offset < input_grid.shape[1]:
                if input_grid[x + x_offset, y + y_offset]!= Color.BLACK:
                    # Check if this pixel is symmetric to the black pixel
                    for sym in symmetries:
                        symmetric_x, symmetric_y = sym.apply(x, y, iters=1)
                        if (symmetric_x >= 0 and symmetric_x < input_grid.shape[0] and
                            symmetric_y >= 0 and symmetric_y < input_grid.shape[1] and
                            input_grid[symmetric_x, symmetric_y] == input_grid[x + x_offset, y + y_offset]):
                            output_grid[x + x_offset, y + y_offset] = input_grid[x + x_offset, y + y_offset]
                            break

    return output_grid