from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color clustering, radial symmetry

# description:
# In the input, you will see a grid with a central pattern of colored pixels. 
# To make the output, you should find all the connected components in the grid, 
# and for each connected component, create a new grid that radiates outward 
# from the center of that component, filling in the surrounding area with a specific color based on the central color of the component.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find connected components in the grid
    components = find_connected_components(input_grid, monochromatic=False, connectivity=8)

    # Initialize output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    for component in components:
        # Get the color of the component
        color = component[0, 0]

        # Find the bounding box of the component
        x, y, width, height = bounding_box(component)

        # Calculate the center of the component
        center_x, center_y = x + width // 2, y + height // 2

        # Create a radial filling around the center
        for i in range(-2, 3):  # -2 to 2
            for j in range(-2, 3):  # -2 to 2
                if abs(i) + abs(j) <= 2:  # Only fill within a Manhattan distance of 2
                    output_grid[center_x + i, center_y + j] = color

    return output_grid