from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object extraction, color mapping

# description:
# In the input, you will see a grid filled with colored objects. Each object can be of various colors, and they may overlap.
# The task is to identify each object, and create a new output grid where each object is replaced with a corresponding color from a predefined mapping:
# red -> blue, blue -> green, green -> yellow, yellow -> red. Objects that are completely black (transparent) should be ignored.

def transform(input_grid):
    # Define the color mapping
    color_map = {
        Color.RED: Color.BLUE,
        Color.BLUE: Color.GREEN,
        Color.GREEN: Color.YELLOW,
        Color.YELLOW: Color.RED
    }

    # Create an output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Find all connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # Iterate through each object
    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)

        # Crop the object from the input grid
        cropped_object = crop(obj, background=Color.BLACK)

        # Get the color of the object
        original_color = np.unique(cropped_object[cropped_object!= Color.BLACK])[0]

        # Map the color using the defined mapping
        new_color = color_map.get(original_color, original_color)  # Default to original color if not mapped

        # Create a mask for the object and fill it with the new color in the output grid
        mask = cropped_object!= Color.BLACK
        output_grid[x:x+cropped_object.shape[0], y:y+cropped_object.shape[1]][mask] = new_color

    return output_grid