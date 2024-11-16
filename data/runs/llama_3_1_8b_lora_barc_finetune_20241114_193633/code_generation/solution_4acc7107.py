from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object extraction, color correspondence, grid transformation

# description:
# In the input, you will see a grid with a black background and various colored objects. 
# Each object has a distinct color. The output should be a new grid where each object is 
# duplicated and placed in the new grid, but the colors are transformed: 
# the color of each object is replaced by the color of the pixel at its top-left corner in the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Step 1: Find all the connected components (objects) in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    
    # Step 2: Create an output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Step 3: For each object, replace its color with the color of the top-left pixel
    for obj in objects:
        # Get the color of the top-left pixel of the object
        top_left_color = obj[0, 0]
        
        # Crop the object to get its sprite
        sprite = crop(obj, background=Color.BLACK)

        # Get the position of the top-left corner of the object
        x, y = np.argwhere(input_grid == obj[0, 0])[0]

        # Place the object in the output grid, changing its color to the top-left color
        for i in range(sprite.shape[0]):
            for j in range(sprite.shape[1]):
                if sprite[i, j]!= Color.BLACK:
                    output_grid[x + i, y + j] = top_left_color

    return output_grid