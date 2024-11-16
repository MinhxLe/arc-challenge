from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color transformation, symmetry

# description:
# In the input grid, you will see a colored shape represented by colored pixels, and a black background. 
# To create the output grid, rotate the shape 180 degrees around its center point, and replace the color of the rotated pixels with a specified color (e.g., blue).

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Find the bounding box of the colored shape
    x, y, width, height = bounding_box(input_grid!= Color.BLACK)
    
    # Extract the shape from the grid
    shape = input_grid[x:x + width, y:y + height]
    
    # Rotate the shape 180 degrees
    rotated_shape = np.rot90(shape, 2)
    
    # Get the color of the original shape
    original_color = shape[shape!= Color.BLACK][0]  # Get the first non-black color

    # Define the color to replace the rotated shape
    replacement_color = Color.BLUE  # This can be parameterized as needed

    # Replace the rotated shape in the output grid with the specified color
    output_grid[x:x + rotated_shape.shape[0], y:y + rotated_shape.shape[1]] = np.where(rotated_shape!= Color.BLACK, replacement_color, Color.BLACK)

    return output_grid