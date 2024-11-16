from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color filling, symmetry detection, pixel manipulation

# description:
# In the input, you will see a grid with a black background and several colored pixels forming a shape. 
# To make the output, you should detect the color of the shape and fill the entire area of the shape with a new color while keeping the original shape intact.

def transform(input_grid):
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Find the connected components in the input grid
    components = find_connected_components(input_grid, background=Color.BLACK)

    # Iterate through each connected component
    for component in components:
        # Get the color of the component
        color = component[0, 0]  # assuming the component is monochromatic
        # Fill the area of the component with a new color (e.g., Color.RED)
        output_grid[output_grid == color] = Color.RED

    return output_grid