from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern detection, color replacement

# description:
# In the input, you will see a grid with a specific pattern of colored pixels. 
# To make the output, detect the pattern and replace all instances of that pattern with a new color, 
# while keeping the background intact.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to modify
    output_grid = np.copy(input_grid)

    # Find connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True)

    # For each detected object, replace it with a new color
    for obj in objects:
        # Get the color of the current object
        object_color = np.unique(obj[obj!= Color.BLACK])[0]
        
        # Replace all pixels of the object's color with a new color (let's say Color.RED)
        output_grid[output_grid == object_color] = Color.RED

    return output_grid