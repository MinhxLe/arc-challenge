from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, symmetry, pattern recognition

# description:
# In the input, you will see a grid with a central symmetric pattern surrounded by a border of colored pixels. 
# The central pattern is made of one color, while the border is a mix of other colors. 
# To create the output, you should replace the central pattern with a new color while keeping the border unchanged. 
# Additionally, if the central pattern is symmetric, change the color of the border pixels to a different color.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to work on
    output_grid = np.copy(input_grid)

    # Detect the bounding box of the central pattern
    objects = find_connected_components(output_grid, background=Color.BLACK, monochromatic=False)
    central_pattern = None

    for obj in objects:
        if np.count_nonzero(obj!= Color.BLACK) > 0:
            central_pattern = obj
            break

    if central_pattern is not None:
        # Get the bounding box of the central pattern
        x, y, width, height = bounding_box(central_pattern, background=Color.BLACK)
        
        # Determine the color of the central pattern
        pattern_color = central_pattern[0, 0]  # Assuming the entire pattern has the same color

        # Replace the central pattern with a new color
        new_color = Color.BLUE if pattern_color!= Color.BLUE else Color.RED  # Choose a color different from the original
        output_grid[x:x + height, y:y + width] = new_color

        # Check for symmetry
        if np.array_equal(central_pattern, np.flip(central_pattern, axis=0)) or np.array_equal(central_pattern, np.flip(central_pattern, axis=1)):
            # Change border colors to a different color (for simplicity, using a fixed color, e.g., Color.GREEN)
            border_color = Color.GREEN
            output_grid[0, :] = border_color  # Top border
            output_grid[-1, :] = border_color  # Bottom border
            output_grid[:, 0] = border_color  # Left border
            output_grid[:, -1] = border_color  # Right border

    return output_grid