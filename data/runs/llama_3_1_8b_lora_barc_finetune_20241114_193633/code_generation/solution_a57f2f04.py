from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, connected components, pixel transformation

# description:
# In the input, you will see a grid with several colored pixels and a black background. 
# To create the output, map the colors of the pixels to a new color scheme where each color is replaced by a different color based on a predefined mapping. 
# The mapping is defined as follows: 
# blue -> red, green -> yellow, yellow -> blue, red -> green, and all other colors remain unchanged.

def transform(input_grid):
    # Initialize output grid
    output_grid = np.copy(input_grid)

    # Create a color mapping
    color_map = {
        Color.BLUE: Color.RED,
        Color.GREEN: Color.YELLOW,
        Color.YELLOW: Color.BLUE,
        Color.RED: Color.GREEN
    }

    # Apply the color mapping
    output_grid = np.vectorize(lambda color: color_map.get(color, color))(output_grid)

    return output_grid