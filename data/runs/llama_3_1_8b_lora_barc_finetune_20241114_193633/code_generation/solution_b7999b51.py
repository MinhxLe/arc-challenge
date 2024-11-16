from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# object separation, color filling

# description:
# In the input, you will see a grid filled with various colored objects. Each object is a connected component of the same color.
# To create the output, you should separate each object into its own grid and fill the background of each object's grid with a specified color.
# The output grid should contain all the separated objects arranged in a 1xN grid format, where N is the number of objects.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Step 1: Find all connected components (objects) in the input grid
    objects = find_connected_components(input_grid, monochromatic=False, connectivity=4)

    # Step 2: Prepare the output grid
    num_objects = len(objects)
    output_grid = np.zeros((num_objects, input_grid.shape[1]), dtype=int)  # Each object gets a 1xN grid

    # Step 3: Place each object into the output grid
    for i, obj in enumerate(objects):
        # Crop the object to its bounding box
        sprite = crop(obj, background=Color.BLACK)
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj)
        
        # Step 4: Create a new grid for the current object, filling it with the background color
        output_grid[i, :] = Color.BLACK  # Fill with background color (black)

        # Blit the sprite onto the output grid
        blit_sprite(output_grid[i, :], sprite, background=Color.BLACK)

    return output_grid