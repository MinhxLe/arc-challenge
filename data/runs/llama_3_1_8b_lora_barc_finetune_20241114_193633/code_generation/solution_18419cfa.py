from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, layering, transparency

# description:
# In the input, you will see a grid with a colored object and a semi-transparent overlay pattern.
# To make the output, blend the colors of the object with the overlay pattern, creating a new color for each pixel based on the overlapping colors.

def transform(input_grid):
    # Create an output grid initialized to the background color
    output_grid = np.copy(input_grid)

    # Find the connected components of the colored object
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True)
    
    # Assume the first object found is the colored object
    colored_object = objects[0]

    # Create a mask for the colored object
    colored_mask = np.where(colored_object!= Color.BLACK, 1, 0)

    # Define the overlay pattern (for example, a checkerboard pattern)
    overlay_pattern = np.array([[Color.BLACK, Color.RED, Color.BLACK, Color.RED],
                               [Color.RED, Color.BLACK, Color.BLACK, Color.BLACK],
                               [Color.BLACK, Color.RED, Color.BLACK, Color.RED]])

    # Get the dimensions of the overlay pattern
    overlay_height, overlay_width = overlay_pattern.shape

    # Iterate through the grid to blend colors
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if colored_mask[x, y] == 1:
                # If there's a colored pixel, apply the overlay pattern
                for dx in range(overlay_width):
                    for dy in range(overlay_height):
                        # Calculate the position for the overlay pattern
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < input_grid.shape[0] and 0 <= ny < input_grid.shape[1]:
                            # Blend colors if the overlay pixel is not the background
                            if overlay_pattern[dy, dx]!= Color.BLACK:
                                # Simple average blending: mixing the colors
                                output_grid[nx, ny] = blend_colors(output_grid[nx, ny], overlay_pattern[dy, dx])

    return output_grid

def blend_colors(color1, color2):
    # Simple blending function that mixes two colors. For simplicity, we assume colors are represented as integers.
    if color1 == Color.BLACK:
        return color2
    if color2 == Color.BLACK:
        return color1
    # This can be improved with a more sophisticated blending method
    return Color.RED  # Placeholder for blending logic