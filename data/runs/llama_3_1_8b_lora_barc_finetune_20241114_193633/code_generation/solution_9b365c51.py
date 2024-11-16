from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color mapping, counting, horizontal and vertical grouping

# description:
# In the input, you will see a grid filled with colored pixels, where each color appears multiple times.
# The output should consist of a grid where each unique color is represented by a block of pixels equal to the count of its occurrences in the input grid.
# The blocks should be arranged in the output grid in horizontal and vertical order of their first appearance in the input.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a dictionary to count occurrences of each color
    color_counts = {}
    
    # Count the occurrences of each color
    for color in Color.ALL_COLORS:
        count = np.count_nonzero(input_grid == color)
        if count > 0:
            color_counts[color] = count

    # Create the output grid
    total_colors = len(color_counts)
    max_length = max(color_counts.values())
    output_grid = np.full((total_colors, max_length), Color.BLACK)

    # Fill the output grid with the counted colors
    for idx, (color, count) in enumerate(color_counts.items()):
        output_grid[idx, :count] = color

    return output_grid