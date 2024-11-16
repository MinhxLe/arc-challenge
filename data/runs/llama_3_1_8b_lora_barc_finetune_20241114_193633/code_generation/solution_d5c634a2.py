from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# object extraction, symmetry, color mapping

# description:
# In the input, you will see a grid with a central colored object and a black background. 
# The object can have various colors. 
# To create the output, you should extract the object and mirror it along the vertical axis,
# filling the mirrored section with a specified color (e.g., blue).

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Step 1: Crop the object from the grid
    object_sprite = crop(input_grid, background=Color.BLACK)

    # Step 2: Create an output grid
    output_grid = np.full(object_sprite.shape, Color.BLACK, dtype=int)

    # Step 3: Create the mirrored version of the object
    mirrored_sprite = np.flip(object_sprite, axis=1)

    # Step 4: Determine the position to place the mirrored sprite
    # Assuming the object is placed in the center of the output grid
    output_grid = blit_sprite(output_grid, object_sprite, x=0, y=0, background=Color.BLACK)
    blit_sprite(output_grid, mirrored_sprite, x=0, y=0, background=Color.BLACK)

    # Step 5: Fill the mirrored section with a specified color (e.g., blue)
    output_grid[0:object_sprite.shape[0], 0:object_sprite.shape[1]] = Color.BLUE

    return output_grid