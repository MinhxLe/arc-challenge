from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# counting, extraction, color transformation

# description:
# In the input, you will see a grid with several colored stripes arranged vertically. 
# To create the output, count the number of stripes of each color and create a new grid 
# that contains the colors of the stripes in the order of their count, starting from the top.
# Each stripe will be represented by a single pixel of its color in the output grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Count occurrences of each color in vertical stripes
    color_counts = {}
    for x in range(input_grid.shape[0]):
        color = input_grid[x, 0]
        if color!= Color.BLACK:  # Only consider non-background colors
            if color not in color_counts:
                color_counts[color] = 0
            color_counts[color] += 1

    # Sort colors by their counts in descending order
    sorted_colors = sorted(color_counts.items(), key=lambda item: item[1], reverse=True)
    output_colors = [color for color, count in sorted_colors]

    # Create output grid with dimensions based on the number of unique colors
    output_grid = np.zeros((len(output_colors), input_grid.shape[1]), dtype=int)

    # Fill the output grid with the counted colors
    for i, color in enumerate(output_colors):
        draw_line(output_grid, i, 0, length=input_grid.shape[1], color=color)

    return output_grid