from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, color mapping, arrangement

# description:
# In the input, you will see a grid with several colored objects on a black background. 
# Each object is a connected component of a single color. The output grid should be a new grid 
# where each object is represented by its color, arranged in a row starting from the top-left corner, 
# and the colors are mapped based on their original color. The mapping is as follows:
# red -> blue, blue -> green, green -> yellow, yellow -> red.

def transform(input_grid):
    # Initialize output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Detect all colored objects in the input grid
    objects = detect_objects(input_grid, monochromatic=True, connectivity=4)

    # Create a color mapping
    color_map = {
        Color.RED: Color.BLUE,
        Color.BLUE: Color.GREEN,
        Color.GREEN: Color.YELLOW,
        Color.YELLOW: Color.RED,
    }

    # Draw each object in the output grid according to the mapping
    for obj in objects:
        # Get the color of the object
        color = obj[obj!= Color.BLACK][0]  # Get the color of the first non-black pixel
        # Apply color mapping
        new_color = color_map.get(color, Color.BLACK)  # Default to black if not in mapping
        # Find the bounding box of the object
        x, y, w, h = bounding_box(obj)
        
        # Blit the object into the output grid with the new color
        blit_sprite(output_grid, obj, x=x, y=y, background=Color.BLACK)
        
        # Fill the bounding box area with the new color
        output_grid[x:x+w, y:y+h] = new_color

    return output_grid