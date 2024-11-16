from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern recognition, color mapping, grid transformation

# description:
# In the input, you will see a grid with a specific pattern of colors arranged in a rectangular shape.
# To create the output, you should find the largest rectangular region of the same color in the input grid,
# and then fill the entire grid with that color, while keeping the original grid's structure.

def transform(input_grid):
    # Detect the connected components in the grid
    components = find_connected_components(input_grid, monochromatic=False, connectivity=4)
    
    # Find the largest connected component by area (number of pixels)
    largest_component = max(components, key=lambda comp: np.sum(comp!= Color.BLACK))
    
    # Get the color of the largest component
    largest_color = largest_component[largest_component!= Color.BLACK][0]  # Get the first non-black color

    # Create the output grid filled with the largest color
    output_grid = np.full(input_grid.shape, largest_color)

    return output_grid