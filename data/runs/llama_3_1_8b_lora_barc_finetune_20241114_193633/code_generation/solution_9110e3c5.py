from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel counting, color mapping

# description:
# In the input, you will see a grid with various colored pixels. Each color represents a different type of fruit.
# To make the output, count the number of each fruit type and create a new grid where each fruit type is represented by a specific color:
# - Apple: Color.RED
# - Banana: Color.BLUE
# - Grape: Color.GREEN
# - Watermelon: Color.YELLOW
# - Peach: Color.PINK
# - Orange: Color.ORANGE
# - The number of each fruit type should be represented by the count of that color in the output grid.

def transform(input_grid):
    # Initialize a dictionary to count the occurrences of each fruit type
    fruit_count = {
        Color.RED: 0,  # Apple
        Color.BLUE: 0,  # Banana
        Color.GREEN: 0,  # Grape
        Color.YELLOW: 0,  # Watermelon
        Color.PINK: 0,  # Peach
        Color.ORANGE: 0  # Orange
    }

    # Count occurrences of each fruit type
    for x, y in np.argwhere(input_grid!= Color.BLACK):
        if input_grid[x, y] in fruit_count:
            fruit_count[input_grid[x, y]] += 1

    # Create the output grid based on counts
    output_grid = np.zeros((3, 3), dtype=int)  # Fixed size for simplicity

    # Fill the output grid with colors based on counts
    for color, count in fruit_count.items():
        # Limit to a maximum of 3 fruits per type for simplicity
        for i in range(min(count, 3)):
            output_grid[i // 3, i % 3] = color  # Place colors in a 3x3 grid

    return output_grid