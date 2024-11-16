from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# merging, color transformation

# description:
# In the input, you will see two colored shapes on a black background. 
# To create the output grid, merge these two shapes into one by combining their colors into a new color, 
# and place the merged shape in the center of the output grid.

def transform(input_grid):
    # Detect the two shapes in the input grid
    shapes = detect_objects(grid=input_grid, colors=Color.NOT_BLACK, monochromatic=False, connectivity=4)
    
    # Ensure we have exactly two shapes
    assert len(shapes) == 2
    
    # Create an output grid
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Create a merged color by averaging the colors of the two shapes
    shape1_color = np.unique(shapes[0][shapes[0]!= Color.BLACK])
    shape2_color = np.unique(shapes[1][shapes[1]!= Color.BLACK])
    
    if len(shape1_color) == 1 and len(shape2_color) == 1:
        merged_color = shape1_color[0] if shape1_color[0]!= shape2_color[0] else Color.BLACK
    else:
        merged_color = Color.RED if Color.RED in shape1_color or Color.RED in shape2_color else Color.GREEN
    
    # Place the merged shape in the center of the output grid
    merged_shape = np.full_like(shapes[0], merged_color)
    x_offset = (output_grid.shape[0] - merged_shape.shape[0]) // 2
    y_offset = (output_grid.shape[1] - merged_shape.shape[1]) // 2
    blit_sprite(output_grid, merged_shape, x=x_offset, y=y_offset)

    return output_grid