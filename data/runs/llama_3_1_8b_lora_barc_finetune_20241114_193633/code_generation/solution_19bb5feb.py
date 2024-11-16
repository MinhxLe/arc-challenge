from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# object counting, color mapping, counting

# description:
# In the input, you will see a grid containing several colored objects. Each object is represented by a distinct color, and the background is black.
# To make the output, count the number of objects of each color and represent these counts as colored squares in the output grid.
# Each square should be placed in a grid format, where the number of rows and columns corresponds to the number of colors counted.
# The colors of the squares in the output grid should correspond to the colors of the counted objects.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Plan:
    # 1. Extract the objects from the input grid.
    # 2. Count the number of objects of each color.
    # 3. Create an output grid representing the counts as colored squares.

    # Find connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    color_counts = {}
    for obj in objects:
        color = obj[0, 0]  # Get the color of the object (assuming it's monochromatic)
        if color not in color_counts:
            color_counts[color] = 0
        color_counts[color] += 1

    # Determine the number of colors and prepare the output grid
    num_colors = len(color_counts)
    output_grid = np.zeros((num_colors, num_colors), dtype=int)

    # Fill the output grid with the color counts
    for i, (color, count) in enumerate(color_counts.items()):
        for j in range(count):
            if j < num_colors:  # Ensure we don't exceed the grid size
                output_grid[i, j] = color

    return output_grid