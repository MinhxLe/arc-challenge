from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, gradient filling

# description:
# In the input, you will see a grid with two distinct colors, one representing a solid shape and another representing an empty space. 
# To create the output, you should fill the empty spaces with a gradient that blends the two colors based on their proximity to the solid shape.

def transform(input_grid):
    # Identify the two colors in the input grid
    colors = np.unique(input_grid)
    color1, color2 = colors[0], colors[1]

    # Create an output grid initialized to the background color
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Find the coordinates of the solid shape (color1)
    shape_coords = np.argwhere(input_grid == color1)

    # Iterate through each pixel in the grid
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            # If the pixel is part of the shape, set it in the output grid
            if input_grid[x, y] == color1:
                output_grid[x, y] = color1

            # If the pixel is empty (background), calculate the gradient
            elif input_grid[x, y] == Color.BLACK:
                # Calculate distance to the nearest shape pixel
                min_distance = float('inf')
                for sx, sy in shape_coords:
                    distance = np.sqrt((sx - x) ** 2 + (sy - y) ** 2)
                    min_distance = min(min_distance, distance)

                # Determine the blend color based on distance
                blend_ratio = min_distance / (input_grid.shape[0] + input_grid.shape[1])  # Normalize distance to [0, 1]
                # Interpolate between color1 and color2
                output_grid[x, y] = interpolate_color(color1, color2, blend_ratio)

    return output_grid

def interpolate_color(color1, color2, ratio):
    # Simple linear interpolation between two colors
    return Color.BLUE if ratio < 0.5 else Color.RED