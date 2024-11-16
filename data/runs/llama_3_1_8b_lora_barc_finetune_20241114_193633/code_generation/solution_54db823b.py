from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape detection, color mapping

# description:
# In the input, you will see a grid with various shapes in different colors. 
# To make the output, replace each shape's color with the color of the nearest pixel of a different color in the same row or column.

def transform(input_grid):
    # Create a copy of the input grid to store the output
    output_grid = np.copy(input_grid)

    # Get the dimensions of the grid
    rows, cols = output_grid.shape

    # Iterate through each pixel in the grid
    for x in range(rows):
        for y in range(cols):
            # Get the current color
            current_color = output_grid[x, y]
            if current_color == Color.BLACK:
                continue  # Skip black pixels

            # Check for the nearest different color in the row
            nearest_color_row = None
            nearest_color_col = None
            min_distance_row = float('inf')
            min_distance_col = float('inf')

            # Check row to the left
            for j in range(y - 1, -1, -1):
                if output_grid[x, j]!= Color.BLACK and output_grid[x, j]!= current_color:
                    if output_grid[x, j]!= Color.BLACK:
                        if abs(y - j) < min_distance_row:
                            min_distance_row = abs(y - j)
                            nearest_color_row = output_grid[x, j]

            # Check row to the right
            for j in range(y + 1, cols):
                if output_grid[x, j]!= Color.BLACK and output_grid[x, j]!= current_color:
                    if abs(y - j) < min_distance_row:
                        min_distance_row = abs(y - j)
                        nearest_color_row = output_grid[x, j]

            # Check column above
            for i in range(x - 1, -1, -1):
                if output_grid[i, y]!= Color.BLACK and output_grid[i, y]!= current_color:
                    if abs(x - i) < min_distance_col:
                        min_distance_col = abs(x - i)
                        nearest_color_col = output_grid[i, y]

            # Check column below
            for i in range(x + 1, rows):
                if output_grid[i, y]!= Color.BLACK and output_grid[i, y]!= current_color:
                    if abs(x - i) < min_distance_col:
                        min_distance_col = abs(x - i)
                        nearest_color_col = output_grid[i, y]

            # Set the output grid's color to the nearest different color found
            if nearest_color_row is not None:
                output_grid[x, y] = nearest_color_row
            elif nearest_color_col is not None:
                output_grid[x, y] = nearest_color_col

    return output_grid