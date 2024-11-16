from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color propagation, pixel transformation, connectivity

# description:
# In the input, you will see a grid with a mixture of colored pixels. Each pixel can be one of the 10 colors. 
# To make the output, for each pixel that is red, change all connected pixels of the same color to a new color (e.g., blue) 
# and for each pixel that is blue, change all connected pixels of the same color to another color (e.g., green). 
# The connectivity is 4-way, and the transformation should not overwrite other colors.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Define the new colors for the transformations
    red_color = Color.RED
    blue_color = Color.BLUE
    red_output_color = Color.BLUE
    blue_output_color = Color.GREEN

    # Find all connected components of red and blue pixels
    red_components = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True, connectivity=4)
    blue_components = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True, connectivity=4)

    # Change red components to red_output_color
    for component in red_components:
        output_grid[component == red_color] = red_output_color

    # Change blue components to blue_output_color
    for component in blue_components:
        output_grid[component == blue_color] = blue_output_color

    return output_grid