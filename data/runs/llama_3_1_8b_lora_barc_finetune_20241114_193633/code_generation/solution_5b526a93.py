from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape detection, color replacement

# description:
# In the input, you will see a grid with various colored shapes and a background. The task is to identify the shapes that are 
# contiguous and replace all their colors with a specific color (e.g., teal), while leaving the background intact.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Find connected components (shapes) in the grid
    shapes = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    
    for shape in shapes:
        # Get the bounding box of each shape
        x, y, width, height = bounding_box(shape, background=Color.BLACK)
        
        # Replace the shape's color with teal
        output_grid[y:y + height, x:x + width] = Color.TEAL

    return output_grid