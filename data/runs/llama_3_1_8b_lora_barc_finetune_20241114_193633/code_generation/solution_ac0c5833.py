from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling

# description:
# In the input, you will see a grid with a symmetrical pattern and a few scattered colored pixels. 
# To create the output, fill in the missing parts of the pattern to complete the symmetry while ensuring that the filled colors match the existing colors in the pattern.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to modify for output
    output_grid = np.copy(input_grid)
    
    # Find all connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True, connectivity=4)

    # Fill in the missing parts of the pattern to complete the symmetry
    for obj in objects:
        # Get the bounding box of the current object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)
        
        # Calculate the center of the bounding box
        center_x, center_y = x + width // 2, y + height // 2
        
        # Determine the symmetry type based on the size of the object
        if width == height:  # Square object
            # Check if the object is symmetrical
            if np.array_equal(obj, np.flipud(obj)):
                continue  # Already symmetrical
            # Fill the missing parts to complete symmetry
            for i in range(height):
                for j in range(width):
                    if obj[i, j]!= Color.BLACK:
                        # Fill the corresponding symmetrical position
                        output_grid[height - 1 - i, j] = obj[i, j]  # Vertical symmetry
                        output_grid[i, width - 1 - j] = obj[i, j]  # Horizontal symmetry
                        output_grid[height - 1 - i, width - 1 - j] = obj[i, j]  # Diagonal symmetry
                        output_grid[height - 1 - i, j] = obj[i, j]  # Bottom symmetry
                        output_grid[i, width - 1 - j] = obj[i, j]  # Right symmetry

        else:  # Rectangular object
            # Check horizontal symmetry
            for i in range(height):
                for j in range(width):
                    if obj[i, j]!= Color.BLACK:
                        # Fill the corresponding symmetrical position
                        output_grid[height - 1 - i, j] = obj[i, j]  # Vertical symmetry

            # Check vertical symmetry
            for i in range(height):
                for j in range(width):
                    if obj[i, j]!= Color.BLACK:
                        # Fill the corresponding symmetrical position
                        output_grid[i, width - 1 - j] = obj[i, j]  # Horizontal symmetry

    return output_grid