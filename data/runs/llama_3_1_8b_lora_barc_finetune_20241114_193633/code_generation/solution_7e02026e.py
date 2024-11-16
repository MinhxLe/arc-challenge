from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# distance measurement, color mapping, object detection

# description:
# In the input, you will see a grid with several colored objects, some of which may overlap.
# To make the output, calculate the distance of each object from a central point (the center of the grid) 
# and color them based on their distance:
# The closest objects will be colored green, the next closest will be colored orange, 
# and the farthest will be colored red.

def transform(input_grid):
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)
    
    # Get the dimensions of the grid
    height, width = output_grid.shape
    
    # Calculate the center coordinates
    center_x, center_y = height // 2, width // 2
    
    # Find the coordinates of all colored pixels
    colored_pixels = np.argwhere(output_grid!= Color.BLACK)
    
    # For each colored pixel, calculate the distance from the center
    for x, y in colored_pixels:
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # Determine the color based on distance
        if distance < 2:
            color = Color.GREEN
        elif distance < 4:
            color = Color.ORANGE
        else:
            color = Color.RED
        
        # Fill the pixel with the determined color
        output_grid[x, y] = color
    
    return output_grid