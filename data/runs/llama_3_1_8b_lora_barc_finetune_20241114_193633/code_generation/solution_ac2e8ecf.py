from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, filling, connectivity

# description:
# In the input, you will see a grid with various colored pixels. The grid contains three distinct colors: 
# red, blue, and teal. To create the output, map the colors as follows:
# red -> blue, blue -> green, teal -> yellow. Fill the entire area connected to each pixel of the original colors
# with the new colors according to the mapping.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Initialize output grid with the same shape as input
    output_grid = np.copy(input_grid)

    # Create the color mapping
    color_map = {
        Color.RED: Color.BLUE,
        Color.BLUE: Color.GREEN,
        Color.TEAL: Color.YELLOW
    }

    # Perform the color mapping and fill connected components
    for color in Color.NOT_BLACK:
        if color in color_map:
            # Find all connected components of the original color
            components = find_connected_components(input_grid == color, monochromatic=True, connectivity=4)
            for component in components:
                # Get the bounding box of the component
                x, y, w, h = bounding_box(component)
                # Create a mask for the area to fill
                fill_mask = np.zeros_like(output_grid)
                fill_mask[x:x+w, y:y+h] = Color.BLACK  # Fill the area with black (background)
                # Fill the area with the new color
                fill_mask[component!= Color.BLACK] = color_map[color]
                # Blit the filled mask onto the output grid
                output_grid = blit_sprite(output_grid, fill_mask, background=Color.BLACK)

    return output_grid