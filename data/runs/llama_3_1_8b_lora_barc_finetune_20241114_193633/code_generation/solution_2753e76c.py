from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object counting, color mapping, size transformation

# description:
# In the input, you will see a grid with various colored objects on a black background.
# To create the output, count the number of objects of each color and transform them into a new grid
# where each color's count is represented by a square of that color in the output grid.
# The output grid will be filled with squares of colors corresponding to the counts of the input objects,
# arranged in descending order of their counts, with each square occupying a 2x2 area in the output grid.

def transform(input_grid):
    # 1. Find all connected components in the input grid
    objects = find_connected_components(grid=input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    
    # 2. Count the number of objects per color
    color_counts = {}
    for obj in objects:
        color = obj[0][0]  # Assuming the object is monochromatic
        if color not in color_counts:
            color_counts[color] = 0
        color_counts[color] += 1

    # 3. Sort colors by count in descending order
    sorted_colors = sorted(color_counts.items(), key=lambda item: item[1], reverse=True)
    
    # 4. Create the output grid
    output_grid = np.full((10, 10), Color.BLACK)  # Fixed output grid size

    # 5. Fill the output grid with squares representing counts
    for i, (color, count) in enumerate(sorted_colors):
        if count > 0:
            # Draw a 2x2 square for each count
            for j in range(2):
                for k in range(2):
                    if i < output_grid.shape[0] and i < output_grid.shape[1]:
                        output_grid[i, j] = color
                        output_grid[i + 1, j] = Color.BLACK
                        output_grid[i, j + 1] = Color.BLACK
                        output_grid[i + 1, j + 1] = Color.BLACK

    return output_grid