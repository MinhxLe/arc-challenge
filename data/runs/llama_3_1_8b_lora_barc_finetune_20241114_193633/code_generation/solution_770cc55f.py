from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# reflection, symmetry detection

# description:
# In the input, you will see a grid with a colored object on one side and a vertical mirror line on the opposite side.
# To make the output, reflect the object across the mirror line, coloring the new pixels with a different color.

def transform(input_grid):
    # Plan:
    # 1. Identify the mirror line (which is assumed to be a vertical line in the middle)
    # 2. Reflect the object across the mirror line
    # 3. Color the reflected object with a different color

    output_grid = np.copy(input_grid)

    # Detect the mirror line (assumed to be vertical)
    mirror_line_x = output_grid.shape[1] // 2

    # Find the object's color and its bounding box
    object_color = None
    object_coordinates = []

    # Identify the color of the object
    for x, y in np.argwhere(output_grid!= Color.BLACK):
        object_color = output_grid[x, y]
        object_coordinates.append((x, y))

    # Reflect the object across the mirror line
    for x, y in object_coordinates:
        reflected_x = x
        reflected_y = mirror_line_x + (mirror_line_x - y)

        # If the reflected position is within bounds and not the background
        if 0 <= reflected_y < output_grid.shape[1]:
            output_grid[reflected_x, reflected_y] = Color.YELLOW  # Color the reflected pixels with a new color

    return output_grid