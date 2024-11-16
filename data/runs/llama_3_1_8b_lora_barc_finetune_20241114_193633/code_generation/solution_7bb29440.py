from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel counting, object extraction, color transformation

# description:
# In the input, you will see a grid filled with various colored pixels, some of which are surrounded by a black border.
# To make the output, count the number of pixels of each color inside the black border and create a new grid that
# represents the count of each color in a bar graph format, where the height of each bar corresponds to the count of the respective color.

def transform(input_grid):
    # Find connected components in the grid, ignoring the background color (black)
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    
    # Initialize a dictionary to count the colors
    color_count = {}
    
    # Count the pixels in each object
    for obj in objects:
        # Get the color of the object
        color = obj[0, 0]
        
        # Count the pixels in the object
        pixel_count = np.sum(obj!= Color.BLACK)
        
        # Update the count dictionary
        if color in color_count:
            color_count[color] += pixel_count
        else:
            color_count[color] = pixel_count
    
    # Create the output grid based on the counts
    output_grid = np.zeros((len(color_count), 1), dtype=int)
    
    # Fill the output grid with the counted colors
    for i, (color, count) in enumerate(color_count.items()):
        output_grid[i, 0] = count
    
    return output_grid