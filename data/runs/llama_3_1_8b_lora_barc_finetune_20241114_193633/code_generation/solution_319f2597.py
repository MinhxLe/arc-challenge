from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel grouping, color separation

# description:
# In the input grid, you will see a mixture of colored pixels. The output should separate these pixels into distinct groups based on their colors.
# Each group should be represented by a single pixel in the output grid that represents the color of that group.
# The output grid will be the same size as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to black (background)
    output_grid = np.zeros_like(input_grid)

    # Detect connected components in the input grid
    components = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False, connectivity=4)

    # For each detected component, determine its color and place it in the output grid
    for component in components:
        # Get the color of the current component
        color = component[0, 0]
        
        # Find the bounding box of the component
        x, y, width, height = bounding_box(component, background=Color.BLACK)

        # Fill the output grid at the corresponding position with the color of the component
        output_grid[y:y + height, x:x + width] = color

    return output_grid