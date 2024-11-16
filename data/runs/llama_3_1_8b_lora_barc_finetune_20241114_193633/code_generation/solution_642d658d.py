from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel counting, color transformation, counting colors

# description:
# In the input, you will see a grid with various colored pixels, some of which are black (background).
# The task is to count the total number of each color in the grid and create an output grid that displays
# the counts for each color, sorted by the count in descending order. If a color has a count of 0, it will not be included in the output grid.
# The output grid will be filled with the colors in the order of their counts, with the top row representing the color counts.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Count the occurrences of each color, excluding the background (black)
    color_counts = {}
    for color in Color.ALL_COLORS:
        count = np.sum(input_grid == color)
        if count > 0:
            color_counts[color] = count

    # Sort colors by their counts in descending order
    sorted_colors = sorted(color_counts.items(), key=lambda item: item[1], reverse=True)

    # Create the output grid
    output_grid = np.zeros((len(sorted_colors), input_grid.shape[1]), dtype=int)
    for i, (color, count) in enumerate(sorted_colors):
        for j in range(count):
            if j < output_grid.shape[0]:  # Ensure we don't exceed the grid size
                output_grid[i, j] = color

    return output_grid