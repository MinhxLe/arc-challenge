from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# distance measurement, color transformation, filling

# description:
# In the input, you will see a grid with a central colored object and some colored pixels scattered around.
# To create the output, color the pixels in the output grid based on the distance from the center of the central object:
# - If the pixel is within 1 pixel distance from the center, color it with the color of the central object.
# - If the pixel is 2 pixels away, color it with a different color.
# - If the pixel is more than 2 pixels away, color it black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Find the center of the central object (the first non-black pixel)
    center_x, center_y = np.where(input_grid!= Color.BLACK)
    if center_x.size == 0:
        return output_grid  # No colored pixels found

    center_x, center_y = center_x[0], center_y[0]

    # Get the color of the central object
    central_color = input_grid[center_x, center_y]

    # Fill the output grid based on distance from the center
    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if distance <= 1:
                output_grid[x, y] = central_color
            elif distance == 2:
                output_grid[x, y] = Color.RED  # Different color for 2 pixels away
            else:
                output_grid[x, y] = Color.BLACK  # More than 2 pixels away, color black

    return output_grid