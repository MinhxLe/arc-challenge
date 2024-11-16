from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# counting, color mapping, grid transformation

# description:
# In the input, you will see a grid filled with colored pixels. Each color represents a different type of fruit (e.g., red for apples, green for bananas).
# To create the output, count the number of each color in the grid and generate a new grid that represents the counts in a 2xN grid format:
# - The first row represents the count of red fruits.
# - The second row represents the count of green fruits.
# - Each count is represented by filling the corresponding number of cells in the output grid with that color.

def transform(input_grid):
    # Count the number of red and green pixels
    red_count = np.count_nonzero(input_grid == Color.RED)
    green_count = np.count_nonzero(input_grid == Color.GREEN)

    # Create the output grid
    output_grid = np.zeros((2, 10), dtype=int)  # 2 rows, 10 columns

    # Fill the output grid with counts
    output_grid[0, :red_count] = Color.RED
    output_grid[1, :green_count] = Color.GREEN

    return output_grid