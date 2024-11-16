from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, reflection, color mapping

# description:
# In the input, you will see a grid filled with various colors, with some colors forming symmetric patterns. 
# The task is to identify the symmetric patterns and replicate them across the grid, ensuring that the output grid
# is filled with the detected symmetries. The output grid should maintain the same dimensions as the input grid.

def transform(input_grid):
    # Create an output grid initialized with the background color
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Find connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True)

    # Iterate over each connected component and its bounding box
    for obj in objects:
        # Get the bounding box of the component
        x, y, width, height = bounding_box(obj)
        
        # Crop the component from the input grid
        sprite = crop(obj, background=Color.BLACK)
        
        # Reflect the sprite across its center
        reflected_sprite = sprite[::-1, :]  # Reflect vertically
        reflected_sprite = reflected_sprite[:, ::-1]  # Reflect horizontally
        
        # Blit the original sprite to the output grid
        blit_sprite(output_grid, sprite, x=x, y=y, background=Color.BLACK)
        
        # Blit the reflected sprite to the output grid
        blit_sprite(output_grid, reflected_sprite, x=x, y=y + height, background=Color.BLACK)

    return output_grid