from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# spiral pattern, color mapping, pixel manipulation

# description:
# In the input grid, you will see a colored object represented by colored pixels.
# To create the output grid, replace each pixel of the object with a spiral pattern that radiates outward from the original position.
# The spiral pattern should alternate colors as it expands outward, using the same color as the original pixel at the center.


def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Find all colored pixels in the input grid
    colored_pixels = np.argwhere(input_grid != Color.BLACK)

    for x, y in colored_pixels:
        # Get the color of the current pixel
        color = input_grid[x, y]

        # Create a spiral pattern around the pixel
        spiral_coords = []
        radius = 0
        while True:
            # Draw the spiral in four quadrants
            for dx in [-1, 0, 1, 0]:  # (-1, 0), (0, 1), (1, 0), (0, -1)
                for dy in [-1, 1]:  # -1, 1
                    if (
                        x + dx >= 0
                        and x < output_grid.shape[0]
                        and y + dy >= 0
                        and y < output_grid.shape[1]
                    ):
                        spiral_coords.append((x + dx, y + dy))
                    if not spiral_coords:
                        break
                radius += 1
                if not spiral_coords:
                    break

        # Place the spiral in the output grid
        for x_, y_ in spiral_coords:
            if 0 <= x_ < output_grid.shape[0] and 0 <= y_ < output_grid.shape[1]:
                output_grid[x_, y_] = color

    return output_grid

