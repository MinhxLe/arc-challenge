from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# layering, color blending, transparency

# description:
# In the input, you will see a colored object on a transparent background.
# To create the output, overlay another colored object on top of the existing one, blending the colors where they overlap using a specified blending rule (e.g., average, overlay, etc.).
# The output grid should show the new layered object.

def transform(input_grid):
    # Plan:
    # 1. Extract the background color
    # 2. Find the object in the input grid
    # 3. Create a new grid for the output
    # 4. Overlay the new object on the existing object
    # 5. Blend colors where the objects overlap

    # 1. Find the background color
    background = np.unique(input_grid)[0]

    # 2. Detect the objects in the input grid
    objects = detect_objects(grid=input_grid, monochromatic=False, background=background, connectivity=4, allowed_dimensions=None)

    # Create an output grid initialized to the background color
    output_grid = np.full(input_grid.shape, background)

    # Overlay each object onto the output grid
    for obj in objects:
        # Determine the color of the current object
        color = obj[0, 0]  # Assuming the object is monochromatic
        obj_coords = np.argwhere(obj!= Color.BLACK)

        # Overlay the object on the output grid
        for x, y in obj_coords:
            if output_grid[x, y] == Color.BLACK:  # Only place it if the position is black
                output_grid[x, y] = color

    # Blend colors where the objects overlap
    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            if output_grid[x, y]!= Color.BLACK:
                # Check if there's another color (not background) in the same position
                if output_grid[x, y]!= Color.BLACK and (x, y) in np.argwhere(input_grid!= Color.BLACK):
                    # Blend colors (for simplicity, we can use a simple rule here)
                    output_grid[x, y] = blend_colors(output_grid[x, y], input_grid[x, y])

    return output_grid


def blend_colors(color1, color2):
    # Simple blending rule: if both colors are not background, return the average color
    if color1 == Color.BLACK or color2 == Color.BLACK:
        return color1
    return Color.RED if (color1 == Color.RED and color2 == Color.BLUE) else color1  # Example blend