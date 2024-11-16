from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# layering, masking

# description:
# In the input, you will see several colored layers stacked on top of each other. The layers are represented by non-black pixels, and the background is black. 
# To make the output, you need to separate each layer into its own grid, masking out the others while keeping their original colors intact.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to the background color (black)
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Detect the layers by finding connected components in the input grid
    layers = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True)

    for layer in layers:
        # Get the bounding box of the layer
        x, y, width, height = bounding_box(layer)
        
        # Crop the layer to get the actual sprite
        sprite = crop(layer)

        # Get the color of the current layer
        color = sprite[0, 0]  # Assuming the sprite is monochromatic
        
        # Blit the sprite onto the output grid at the original position, masking out the background
        blit_sprite(output_grid, sprite, x, y, background=Color.BLACK)

    return output_grid