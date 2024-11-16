from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# fractal patterns, recursive filling

# description:
# In the input, you will see a simple fractal pattern made up of colored pixels. 
# To create the output, recursively fill in the empty spaces with the same fractal pattern, 
# maintaining the color and structure.

def transform(input_grid):
    # Find the connected components (fractal patterns) in the input grid
    objects = find_connected_components(input_grid, monochromatic=False, connectivity=8)

    # Determine the bounding box of the first detected object
    fractal_object = objects[0]
    x, y, width, height = bounding_box(fractal_object)

    # Create an output grid larger than the input grid to accommodate the fractal filling
    output_grid = np.full((height * 3, width * 3), Color.BLACK)

    # Place the original fractal pattern in the center of the output grid
    blit_sprite(output_grid, fractal_object, x=width, y=height, background=Color.BLACK)

    # Fill the output grid with the fractal pattern in a recursive manner
    def fill_fractal(x, y, scale):
        for dx in range(-scale, scale + 1):
            for dy in range(-scale, scale + 1):
                if abs(dx) + abs(dy) == scale:  # Check if within the boundary of the fractal
                    # Calculate the position to blit the fractal pattern
                    blit_sprite(output_grid, fractal_object, x + dx, y + dy, background=Color.BLACK)

    # Fill the surrounding area with the fractal pattern
    for scale in range(1, 3):  # Fill up to 2 scales around the center
        for dx in range(-scale, scale + 1):
            for dy in range(-scale, scale + 1):
                fill_fractal(x + dx, y + dy, scale)

    return output_grid