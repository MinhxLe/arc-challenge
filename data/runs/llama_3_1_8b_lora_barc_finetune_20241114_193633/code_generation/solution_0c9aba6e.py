from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, boundary detection

# description:
# In the input, you will see a grid with a colored shape surrounded by a border of a different color. 
# The shape consists of various colors, and the border color is uniform. 
# To make the output, replace the border color with a new color while keeping the shape intact, 
# and fill the area outside the shape with a different color.

def transform(input_grid):
    # Detect the border color (the color that surrounds the shape)
    border_color = None
    for color in Color.NOT_BLACK:
        if np.all(input_grid[0, :] == color) or np.all(input_grid[:, 0] == color):
            border_color = color
            break
    
    # Create a new output grid initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Find the connected components of the shape
    shape_components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    
    # Fill the output grid with the shape
    for shape in shape_components:
        # Crop the shape to remove any surrounding black pixels
        sprite = crop(shape, background=Color.BLACK)
        # Determine the position to blit the shape into the output grid
        x, y, w, h = bounding_box(shape, background=Color.BLACK)
        blit_sprite(output_grid, sprite, x=x, y=y, background=Color.BLACK)

    # Replace the border color with a new color (Color.BLUE in this case)
    output_grid[output_grid == border_color] = Color.BLUE

    return output_grid