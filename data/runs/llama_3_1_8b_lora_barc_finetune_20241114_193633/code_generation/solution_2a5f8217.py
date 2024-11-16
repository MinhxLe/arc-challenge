from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape detection, color transformation, connected components

# description:
# In the input, you will see a grid with various shapes of different colors. 
# To create the output, identify all the shapes, and change their colors based on their 
# top-left position: if a shape is located in the upper half of the grid, change its color 
# to green; if in the lower half, change it to blue.

def transform(input_grid):
    # Find all connected components in the input grid
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Iterate through each component and change its color based on its position
    for component in components:
        # Get the bounding box of the current component
        x, y, width, height = bounding_box(component, background=Color.BLACK)
        
        # Determine the color of the current component
        component_color = component[0, 0]  # Assuming the component is monochromatic

        # Determine if the component is in the upper or lower half
        if y + height / 2 < output_grid.shape[1]:  # Upper half
            output_grid[output_grid == component_color] = Color.GREEN
        else:  # Lower half
            output_grid[output_grid == component_color] = Color.BLUE

    return output_grid