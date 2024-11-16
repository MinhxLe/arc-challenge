from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape detection, filling, color mapping

# description:
# In the input, you will see a grid with several colored shapes on a black background.
# To make the output, identify the largest shape and fill it with a new color while keeping the other shapes unchanged.

def transform(input_grid):
    # Step 1: Find all connected components in the input grid
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    
    # Step 2: Identify the largest component
    largest_component = max(components, key=lambda obj: np.sum(obj!= Color.BLACK))

    # Step 3: Create an output grid initialized to the background color
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Step 4: Fill the largest component with a new color (e.g., Color.PINK)
    new_color = Color.PINK
    output_grid[largest_component!= Color.BLACK] = new_color

    return output_grid