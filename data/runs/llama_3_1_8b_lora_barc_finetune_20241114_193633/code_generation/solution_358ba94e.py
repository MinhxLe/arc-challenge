from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color isolation, background extraction

# description:
# In the input, you will see a grid with a colored object on a black background.
# To make the output, extract the colored object from the grid, leaving only the pixels of that color,
# while removing all other pixels, effectively isolating the object.

def transform(input_grid):
    # Create an output grid filled with the background color
    output_grid = np.full(input_grid.shape, Color.BLACK, dtype=int)

    # Get the unique color in the input grid (the color of the object)
    unique_colors = np.unique(input_grid)
    object_color = unique_colors[unique_colors!= Color.BLACK][0]

    # Set the pixels of the object color in the output grid
    output_grid[input_grid == object_color] = object_color

    # Crop the output grid to remove any background (black) pixels around the object
    output_grid = crop(output_grid, background=Color.BLACK)

    return output_grid