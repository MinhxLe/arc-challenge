from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern detection, color transformation, grid manipulation

# description:
# In the input, you will see a grid containing two distinct patterns: one for the red pixels and one for the blue pixels. 
# To create the output, you need to identify the two patterns and replace each instance of the red pattern with a green circle and each instance of the blue pattern with a yellow square. 
# The output grid should be the same size as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    red_pattern = detect_objects(grid=input_grid, colors=[Color.RED], connectivity=4, monochromatic=True, allowed_dimensions=[(3, 3)])[0]
    blue_pattern = detect_objects(grid=input_grid, colors=[Color.BLUE], connectivity=4, monochromatic=True, allowed_dimensions=[(3, 3)])[0]
    
    # Draw green circles for red patterns
    for x, y in np.argwhere(red_pattern!= Color.BLACK):
        output_grid[x, y] = Color.GREEN

    # Draw yellow squares for blue patterns
    for x, y in np.argwhere(blue_pattern!= Color.BLACK):
        output_grid[x, y] = Color.YELLOW
    
    return output_grid