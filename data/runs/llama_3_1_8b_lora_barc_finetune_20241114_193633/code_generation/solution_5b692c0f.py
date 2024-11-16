from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color filling, boundary tracing, object detection

# description:
# In the input, you will see a grid with a colored shape surrounded by a black border. 
# To create the output grid, you should fill in the area enclosed by the black border with the color of the shape, effectively creating a solid colored object.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Find the background color (which is black in this case)
    background = Color.BLACK

    # Find the connected components (the colored shape)
    shape_objects = find_connected_components(input_grid, background=background, connectivity=4, monochromatic=True)

    # There should be only one object (the colored shape)
    assert len(shape_objects) == 1
    shape = shape_objects[0]

    # Get the color of the shape
    shape_color = shape[shape!= background][0]

    # Get the bounding box of the shape
    x, y, width, height = bounding_box(shape)

    # Fill the area enclosed by the black border with the shape's color
    for i in range(x, x + width):
        for j in range(y, y + height):
            if output_grid[i, j] == background:
                # If it's black, fill it with the shape's color
                output_grid[i, j] = shape_color

    return output_grid