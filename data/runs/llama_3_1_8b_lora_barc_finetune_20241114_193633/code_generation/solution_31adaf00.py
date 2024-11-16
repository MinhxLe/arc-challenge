from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color blending, masking, overlay

# description:
# In the input, you will see a colored object on a black background. 
# To make the output, create a mask of the object and overlay it onto a new grid, 
# blending the colors of the object with the background color wherever the mask is black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Find the bounding box of the colored object
    x, y, width, height = bounding_box(input_grid!= Color.BLACK)
    
    # Extract the object from the input grid
    object_sprite = input_grid[x:x + width, y:y + height]

    # Create a mask for the object (non-black pixels)
    mask = object_sprite!= Color.BLACK

    # Overlay the object onto the output grid, blending with the background
    for i in range(width):
        for j in range(height):
            if mask[i, j]:  # Only blend if the pixel is part of the object
                # If the pixel is black, blend with background
                if object_sprite[i, j]!= Color.BLACK:
                    output_grid[x + i, y + j] = object_sprite[i, j]
                # If the pixel is not black, keep it as is
                # This can be enhanced with a blending algorithm if needed

    return output_grid