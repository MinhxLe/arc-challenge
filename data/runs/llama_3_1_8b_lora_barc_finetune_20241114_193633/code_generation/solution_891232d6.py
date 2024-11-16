from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, gradient filling

# description:
# In the input, you will see a grid filled with colored pixels, and a single colored pixel in the center of the grid.
# To make the output, you should fill the entire grid with a gradient that transitions from the color of the center pixel to the background color, 
# starting from the center pixel and spreading outward.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2
    center_color = output_grid[center_x, center_y]

    # Get the background color
    background_color = Color.BLACK

    # Create a gradient filling function
    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            # Calculate the distance from the center
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_distance = np.sqrt(2) * (output_grid.shape[0] // 2)  # max distance from center to corners
            gradient_value = distance / max_distance  # Normalized distance

            # Determine the color for the current pixel based on the gradient
            if gradient_value < 0.5:
                # Interpolate between center color and background color
                output_grid[x, y] = center_color
            else:
                output_grid[x, y] = background_color

    return output_grid