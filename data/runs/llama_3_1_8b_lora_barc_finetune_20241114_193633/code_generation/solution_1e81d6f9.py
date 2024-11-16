from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# layering, color blending

# description:
# In the input, you will see a grid with several colored layers stacked on top of each other. 
# The topmost layer is the first non-background pixel found in the grid. 
# To create the output, extract the topmost layer, remove the lower layers, and then color the background with the color of the extracted layer.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the topmost layer (the first non-background pixel)
    topmost_layer_color = None
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y]!= Color.BLACK:
                topmost_layer_color = input_grid[x, y]
                break
        if topmost_layer_color:
            break
    
    # Create the output grid initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Fill the output grid with the topmost layer color
    output_grid[input_grid == topmost_layer_color] = topmost_layer_color

    return output_grid