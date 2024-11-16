from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color blending, object detection

# description:
# In the input grid, you will see a black background with various colored objects.
# To create the output grid, for every pixel in the input grid, if it is colored, 
# blend its color with the color of the nearest neighboring pixel (in all four cardinal directions).
# If there are no neighboring colored pixels, the pixel remains unchanged.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Get the shape of the grid
    rows, cols = input_grid.shape

    for x in range(rows):
        for y in range(cols):
            if input_grid[x, y]!= Color.BLACK:  # Only process non-black pixels
                # Initialize variables to track nearest neighbor
                nearest_color = Color.BLACK
                min_distance = float('inf')

                # Check neighboring pixels
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and input_grid[nx, ny]!= Color.BLACK:
                        # Calculate distance and find the nearest colored pixel
                        distance = abs(dx) + abs(dy)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_color = input_grid[nx, ny]

                # Blend the current pixel color with the nearest neighbor's color
                if nearest_color!= Color.BLACK:
                    output_grid[x, y] = blend_colors(input_grid[x, y], nearest_color)

    return output_grid

def blend_colors(color1: str, color2: str) -> str:
    """
    Simple blending function that averages the RGB values of two colors.
    This function assumes colors are represented as strings, and we will just return
    a new color based on a simple blending rule. For simplicity, we will just
    return the second color if the first color is black.
    """
    if color1 == Color.BLACK:
        return color2
    # Here you can define a more complex blending algorithm if needed.
    return color2  # This is a simple example where the second color takes precedence.