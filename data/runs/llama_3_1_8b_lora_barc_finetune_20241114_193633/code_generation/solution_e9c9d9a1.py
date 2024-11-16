from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, object detection, transformation

# description:
# In the input, you will see a grid with a black background and various colored objects. 
# To make the output, identify all colored objects in the grid, and transform them into a new grid 
# where each object's color is mapped to a different color according to a predefined mapping. 
# The output grid should maintain the same dimensions as the input grid.

def transform(input_grid):
    # Initialize output grid
    output_grid = np.copy(input_grid)

    # Detect all colored objects in the input grid
    objects = detect_objects(grid=input_grid, monochromatic=False, background=Color.BLACK, connectivity=4)

    # Create a color mapping
    color_map = {
        Color.BLUE: Color.GREEN,
        Color.RED: Color.YELLOW,
        Color.GREEN: Color.BLUE,
        Color.YELLOW: Color.RED,
        Color.PINK: Color.PURPLE,
        Color.PURPLE: Color.BROWN,
        Color.ORANGE: Color.GRAY,
        Color.GRAY: Color.PINK,
        Color.BLACK: Color.BLACK,
        Color.BROWN: Color.ORANGE
    }

    # Transform each object in the output grid
    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)
        
        # Crop the object from the input grid
        cropped_object = crop(obj, background=Color.BLACK)

        # Change the color of the cropped object based on the mapping
        for i in range(cropped_object.shape[0]):
            for j in range(cropped_object.shape[1]):
                if cropped_object[i, j]!= Color.BLACK:
                    original_color = cropped_object[i, j]
                    transformed_color = color_map.get(original_color, original_color)  # Default to original if not found
                    cropped_object[i, j] = transformed_color

        # Place the transformed object back into the output grid
        blit_sprite(output_grid, cropped_object, x=x, y=y, background=Color.BLACK)

    return output_grid