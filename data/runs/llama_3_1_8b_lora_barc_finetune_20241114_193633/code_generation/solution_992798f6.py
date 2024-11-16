from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# radial symmetry, distance-based coloring

# description:
# In the input, you will see a grid with a central colored pixel and several other colored pixels scattered around it.
# To create the output, color all pixels that are equidistant from the central pixel with the same color as the central pixel.
# The output grid should maintain the same dimensions as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2
    center_color = output_grid[center_x, center_y]

    # Get the distance from the center
    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if distance == int(distance):
                # Color the pixels at this distance with the center color
                output_grid[x, y] = center_color

    return output_grid