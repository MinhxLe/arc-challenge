from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel counting, color mapping, object detection

# description:
# In the input, you will see a grid containing various colored pixels. The output should represent the count of each color in the input grid.
# Each color will be replaced by its corresponding count in the output grid:
# - 1 pixel of a color -> 1 pixel of that color
# - 2 pixels of a color -> 2 pixels of that color
# - 3 pixels of a color -> 3 pixels of that color
#... so on, up to 10 pixels, which will be represented by the color of the pixel itself.
# If a color counts more than 10 pixels, it will be represented by the color itself in the output grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.full(input_grid.shape, Color.BLACK)  # Start with a black output grid

    # Count the occurrences of each color in the input grid
    color_counts = {}
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            color = input_grid[x, y]
            if color!= Color.BLACK:  # Exclude the background
                if color in color_counts:
                    color_counts[color] += 1
                else:
                    color_counts[color] = 1

    # Map counts to output colors
    for color, count in color_counts.items():
        if count <= 10:  # Represent counts up to 10 directly
            output_grid[x, y] = color
        else:  # More than 10 pixels -> Color itself
            output_grid[x, y] = color

    return output_grid