from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, shape detection

# description:
# In the input, you will see a grid with colored shapes (triangles, squares, circles) on a black background. 
# Each shape can be of a different color. 
# To create the output, replace each shape with a new color according to the following mapping:
# triangle -> red, square -> blue, circle -> green. The shapes should retain their positions in the output grid.

def transform(input_grid):
    # Initialize output grid
    output_grid = input_grid.copy()

    # Find connected components (shapes)
    shapes = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False, connectivity=4)

    # Create a mapping for shape colors
    shape_map = {
        Color.RED: Color.RED,    # Triangle
        Color.BLUE: Color.BLUE,  # Square
        Color.GREEN: Color.GREEN # Circle
    }

    # Replace shapes according to the mapping
    for shape in shapes:
        # Get the color of the current shape
        shape_color = shape[0, 0]  # All pixels in the shape have the same color
        # Determine the new color for this shape
        new_color = shape_map.get(shape_color, Color.BLACK)

        # Replace the shape's color in the output grid
        output_grid[output_grid == shape_color] = new_color

    return output_grid