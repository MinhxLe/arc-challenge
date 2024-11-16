from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel transformation, color mapping, boundary detection

# description:
# In the input, you will see a grid with colored pixels forming a shape. 
# To create the output, transform the shape by rotating it 90 degrees clockwise,
# and color the new shape with a unique color based on its size: 
# if the shape's area is less than 5, color it blue; 
# if between 5 and 10, color it green; if greater than 10, color it orange.

def transform(input_grid):
    # Find the bounding box of the shape
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)
    
    # Extract the shape from the input grid
    shape = input_grid[x:x + width, y:y + height]
    
    # Calculate the area of the shape
    area = np.sum(shape!= Color.BLACK)
    
    # Determine the new color based on the area
    if area < 5:
        new_color = Color.BLUE
    elif 5 <= area <= 10:
        new_color = Color.GREEN
    else:
        new_color = Color.ORANGE
    
    # Rotate the shape 90 degrees clockwise
    rotated_shape = np.rot90(shape, -1)
    
    # Create an output grid that is the same size as the original input grid
    output_grid = np.full_like(input_grid, Color.BLACK)
    
    # Blit the rotated shape onto the output grid
    output_grid[x:x + rotated_shape.shape[0], y:y + rotated_shape.shape[1]] = rotated_shape
    
    # Fill the newly created shape with the determined color
    output_grid[output_grid == Color.BLACK] = new_color
    
    return output_grid