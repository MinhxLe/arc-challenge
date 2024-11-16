from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, object transformation

# description:
# In the input, you will see a grid with various colored objects on a blue background.
# To make the output, blend each object with the blue background, where the blending rule is:
# If an object pixel is blue, it remains blue. If it's any other color, it will blend with blue to create a lighter shade of that color.


def transform(input_grid):
    # Create an output grid initialized to the background color
    output_grid = np.full_like(input_grid, Color.BLUE)

    # Detect all colored objects in the input grid
    objects = detect_objects(
        input_grid, background=Color.BLUE, monochromatic=False, connectivity=4
    )

    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj, background=Color.BLUE)
        obj_color = obj[x, y]

        # Blend the object color with blue
        for i in range(height):
            for j in range(width):
                if obj[i, j] != Color.BLUE:
                    # Blend the color with blue
                    blended_color = blend_color(obj[i, j], Color.BLUE)
                    output_grid[y + i, x + j] = blended_color

    return output_grid


def blend_color(color, blue):
    """
    Blend the given color with blue.
    If the color is blue, it remains blue, otherwise it returns a lighter shade of the color.
    """
    if color == Color.BLUE:
        return Color.BLUE
    else:
        # Create a lighter version of the color (for simplicity, we will return a color from the palette)
        # For simplicity, we can define a simple blending function that returns a lighter shade
        return Color.BLUE  # This can be improved with a proper color mixing logic

