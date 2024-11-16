from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern duplication, symmetry, color filling

# description:
# In the input, you will see a small pattern of colored pixels in the center of a grid, surrounded by a black background.
# To make the output, duplicate this pattern to the four quadrants of the grid, maintaining its orientation and color scheme.

def transform(input_grid):
    # Find the connected components (the pattern in the center)
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8, monochromatic=False)
    
    # Assuming there's only one pattern, we take the first detected component
    pattern = objects[0]
    
    # Get the bounding box of the pattern
    x, y, width, height = bounding_box(pattern)
    
    # Crop the pattern to isolate it
    cropped_pattern = crop(pattern, background=Color.BLACK)
    
    # Determine the size of the output grid
    output_height = max(input_grid.shape[0], height * 2)
    output_width = max(input_grid.shape[1], width * 2)
    
    # Create an output grid filled with black
    output_grid = np.full((output_height, output_width), Color.BLACK)
    
    # Place the pattern in all four quadrants
    blit_sprite(output_grid, cropped_pattern, x=0, y=0, background=Color.BLACK)  # Top-left
    blit_sprite(output_grid, cropped_pattern, x=0, y=output_height - height, background=Color.BLACK)  # Bottom-left
    blit_sprite(output_grid, cropped_pattern, x=output_width - width, y=0, background=Color.BLACK)  # Top-right
    blit_sprite(output_grid, cropped_pattern, x=output_width - width, y=output_height - height, background=Color.BLACK)  # Bottom-right
    
    return output_grid