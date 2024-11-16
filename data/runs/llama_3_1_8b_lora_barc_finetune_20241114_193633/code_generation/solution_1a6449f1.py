from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# object extraction, color mapping, boundary filling

# description:
# In the input, you will see a grid filled with various colored objects on a black background. 
# The task is to extract all objects from the grid, but only those that are connected (4-way connectivity).
# The output grid will be the smallest bounding box that contains all the extracted objects, 
# filled with the color of the largest object found.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Step 1: Extract all connected components from the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Step 2: Determine the largest object based on area (number of non-background pixels)
    largest_object = max(objects, key=lambda obj: np.sum(obj!= Color.BLACK), default=None)

    # Step 3: Create the output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Step 4: Fill the output grid with the largest object
    if largest_object is not None:
        # Crop the largest object to get the sprite
        sprite = crop(largest_object, background=Color.BLACK)

        # Get the color of the largest object
        object_color = np.unique(sprite[sprite!= Color.BLACK])[0]
        
        # Determine the position to place the sprite in the output grid
        x, y = np.argwhere(largest_object!= Color.BLACK)[0]

        # Blit the sprite onto the output grid
        output_grid = blit_sprite(output_grid, sprite, x=x, y=y, background=Color.BLACK)

        # Fill the bounding box of the largest object in the output grid with its color
        bounding_box_coords = bounding_box(largest_object, background=Color.BLACK)
        output_grid[bounding_box_coords[1]:bounding_box_coords[1]+bounding_box_coords[3],
                    bounding_box_coords[0]:bounding_box_coords[0]+bounding_box_coords[2]] = object_color

    return output_grid