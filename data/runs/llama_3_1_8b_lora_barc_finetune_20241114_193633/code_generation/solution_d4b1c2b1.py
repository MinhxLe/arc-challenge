from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# scaling, symmetry detection, color propagation

# description:
# In the input, you will see a small colored object on a black background.
# To create the output, detect any symmetrical properties of the object and scale it up
# to fill the entire grid while maintaining its symmetry. The scaling factor should be
# determined based on the object's size and the grid's dimensions.

def transform(input_grid):
    # Step 1: Detect the object in the input grid
    objects = detect_objects(grid=input_grid, monochromatic=False, background=Color.BLACK, connectivity=4, allowed_dimensions=[(1, 1), (2, 2), (3, 3)])
    assert len(objects) == 1, "There should be exactly one object in the input grid."

    # Step 2: Crop the object to analyze its symmetry
    object_sprite = crop(objects[0], background=Color.BLACK)
    
    # Step 3: Check for symmetry in the object
    symmetries = detect_mirror_symmetry(object_sprite, ignore_colors=[Color.BLACK])
    
    # Step 4: Scale the object based on its symmetry
    scale_factor = 3  # Fixed scaling factor for the output grid size
    scaled_object = np.zeros((object_sprite.shape[0] * scale_factor, object_sprite.shape[1] * scale_factor), dtype=int)
    
    for x in range(object_sprite.shape[0]):
        for y in range(object_sprite.shape[1]):
            color = object_sprite[x, y]
            if color!= Color.BLACK:
                # Fill in the scaled position based on the detected symmetry
                for dx in range(scale_factor):
                    for dy in range(scale_factor):
                        # Find the corresponding position in the scaled grid
                        scaled_x = x * scale_factor + dx
                        scaled_y = y * scale_factor + dy
                        # Apply symmetry
                        for sym in symmetries:
                            symmetric_x, symmetric_y = sym.apply(x, y)
                            symmetric_x, symmetric_y = sym.apply(symmetric_x, symmetric_y)
                            if 0 <= symmetric_x < scaled_object.shape[0] and 0 <= symmetric_y < scaled_object.shape[1]:
                                scaled_object[symmetric_x, symmetric_y] = color
    
    # Step 5: Create the output grid
    output_grid = np.full((scaled_object.shape[0], scaled_object.shape[1]), Color.BLACK)
    output_grid = blit_sprite(output_grid, scaled_object, x=0, y=0, background=Color.BLACK)

    return output_grid