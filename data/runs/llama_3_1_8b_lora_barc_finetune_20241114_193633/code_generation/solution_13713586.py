from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, pixel manipulation, filling, mirroring

# description:
# In the input, you will see a grid filled with colored pixels on a black background. The grid contains a black pixel that acts as a mirror. 
# To create the output grid, you should mirror the colored pixels across the black pixel. If a pixel is already on the same position as the black pixel, it should not be mirrored.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Find the position of the black pixel (mirror)
    mirror_position = np.argwhere(input_grid == Color.BLACK)

    # Ensure there's only one black pixel
    if len(mirror_position)!= 1:
        raise ValueError("There should be exactly one black pixel acting as the mirror.")

    mirror_x, mirror_y = mirror_position[0]

    # Find the positions of all colored pixels
    colored_positions = np.argwhere(input_grid!= Color.BLACK)

    # Mirror each colored pixel across the black pixel
    for x, y in colored_positions:
        if (x, y)!= (mirror_x, mirror_y):  # Skip the black pixel
            mirrored_x = 2 * mirror_x - x
            mirrored_y = 2 * mirror_y - y

            # Check if the mirrored position is within bounds
            if 0 <= mirrored_x < input_grid.shape[0] and 0 <= mirrored_y < input_grid.shape[1]:
                output_grid[mirrored_x, mirrored_y] = input_grid[x, y]

    return output_grid