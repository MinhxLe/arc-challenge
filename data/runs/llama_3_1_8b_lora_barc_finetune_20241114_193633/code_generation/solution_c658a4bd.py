from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape extraction, color mapping, boundary detection

# description:
# In the input, you will see a grid filled with various colored shapes, each surrounded by a black border.
# The output should extract the inner shapes, change their colors according to a predefined mapping, and arrange them in a new grid.
# The output grid should be filled with the transformed shapes, maintaining their original arrangement.

def transform(input_grid):
    # Find the connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8, monochromatic=False)

    # Create a mapping of colors to new colors
    color_mapping = {
        Color.RED: Color.BLUE,
        Color.GREEN: Color.YELLOW,
        Color.BLUE: Color.RED,
        Color.YELLOW: Color.GREEN,
        Color.PURPLE: Color.BROWN,
        Color.BROWN: Color.PURPLE,
        Color.PINK: Color.GRAY,
        Color.GRAY: Color.PINK,
        Color.ORANGE: Color.BLACK,
        Color.BLACK: Color.ORANGE
    }

    # Create an output grid that is larger than the input grid to accommodate the shapes
    output_grid = np.full((input_grid.shape[0] + 2, input_grid.shape[1] + 2), Color.BLACK)

    # Place each shape in the output grid with color mapping
    for obj in objects:
        # Get the bounding box of the object
        x, y, w, h = bounding_box(obj, background=Color.BLACK)

        # Crop the object
        sprite = crop(obj, background=Color.BLACK)

        # Apply color mapping
        for i in range(sprite.shape[0]):
            for j in range(sprite.shape[1]):
                if sprite[i, j] in color_mapping:
                    sprite[i, j] = color_mapping[sprite[i, j]]

        # Blit the transformed shape into the output grid
        blit_sprite(output_grid, sprite, x=1, y=1)

    return output_grid