from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# shape extraction, boundary detection

# description:
# In the input you will see a grid with various shapes made of colored pixels. The goal is to extract all shapes 
# that are contiguous (connected components) and create a new grid that contains only these shapes, 
# arranged in a single row. The background color will be black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Step 1: Find all connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Step 2: Create an output grid, which will hold the extracted shapes
    output_grid = np.zeros((len(objects), max(obj.shape[1] for obj in objects)), dtype=int)

    for i, obj in enumerate(objects):
        # Crop the object to get the sprite
        sprite = crop(obj, background=Color.BLACK)
        # Blit the sprite onto the output grid
        blit_sprite(output_grid, sprite, x=i, y=0, background=Color.BLACK)

    return output_grid