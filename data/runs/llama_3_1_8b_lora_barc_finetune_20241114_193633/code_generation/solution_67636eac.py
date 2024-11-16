from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel grouping, color mapping

# description:
# In the input, you will see a grid filled with various colored pixels. The grid contains regions of the same color that are separated by a black background.
# To make the output, you need to group the colored pixels into connected components and create a new grid where each connected component is represented by a distinct color. 
# The output grid should maintain the relative positions of the colored pixels in their respective groups.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to black
    output_grid = np.zeros_like(input_grid)

    # Find connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    
    # Assign a unique color to each connected component
    color_map = {obj: color for obj, color in zip(objects, Color.NOT_BLACK)}
    
    # Place each connected component in the output grid
    for obj, color in color_map.items():
        # Get the bounding box of the object
        x, y, w, h = bounding_box(obj, background=Color.BLACK)
        # Crop the object to its bounding box
        cropped_obj = crop(obj, background=Color.BLACK)
        # Blit the cropped object onto the output grid
        blit_sprite(output_grid, cropped_obj, x, y, background=Color.BLACK)

    return output_grid