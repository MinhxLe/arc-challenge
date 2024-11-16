from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, object detection, transformation

# description:
# In the input, you will see a grid containing various colored objects. Each object can be of different colors, and there will be a single pixel of a different color that serves as a "key" pixel.
# To create the output grid, you should:
# 1. Identify the color of the key pixel.
# 2. Change the color of all pixels in the same connected component (region) as the key pixel to the key color.
# 3. Leave all other pixels unchanged.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    
    # Find the color of the key pixel
    key_pixel_color = None
    for x, y in np.argwhere(input_grid!= Color.BLACK):
        if input_grid[x, y]!= Color.BLACK:
            key_pixel_color = input_grid[x, y]
            break

    # Find the connected component of the key pixel
    key_component = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    
    # Change the color of all pixels in the same component as the key pixel to the key color
    for component in key_component:
        output_grid[component!= Color.BLACK] = key_pixel_color
    
    return output_grid