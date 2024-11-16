from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, scaling, rotation, mirroring

# description:
# In the input, you will see a grid with a colored shape on a black background.
# The shape can be any color from the palette, and it will occupy a rectangular region in the grid.
# To create the output grid, you should scale the shape by a factor of 2, then mirror it horizontally,
# and color the mirrored shape according to the following mapping:
# red -> blue, blue -> green, green -> yellow, yellow -> red, pink -> purple, purple -> brown, brown -> gray, gray -> pink, orange -> orange, black -> black.

def transform(input_grid):
    # Initialize the output grid
    output_grid = np.full((input_grid.shape[0] * 2, input_grid.shape[1] * 2), Color.BLACK)

    # Detect the colored shape
    shape = detect_objects(grid=input_grid, colors=Color.NOT_BLACK, monochromatic=False, connectivity=4, allowed_dimensions=None, colors=None, can_overlap=False)

    # Scale and mirror each detected shape
    for obj in shape:
        # Scale the shape by a factor of 2
        scaled_shape = np.repeat(np.repeat(obj, 2, axis=0), 2, axis=1)

        # Mirror the shape horizontally
        mirrored_shape = scaled_shape[:, ::-1]

        # Map colors according to the specified mapping
        for x in range(scaled_shape.shape[0]):
            for y in range(scaled_shape.shape[1]):
                if scaled_shape[x, y] in color_map:
                    mirrored_shape[x, y] = color_map[scaled_shape[x, y]]

        # Blit the original scaled shape and the mirrored shape onto the output grid
        blit_sprite(output_grid, scaled_shape, x=0, y=0, background=Color.BLACK)  # Top-left
        blit_sprite(output_grid, mirrored_shape, x=scaled_shape.shape[1], y=0, background=Color.BLACK)  # Right
        blit_sprite(output_grid, mirrored_shape, x=0, y=scaled_shape.shape[0], background=Color.BLACK)  # Bottom-left
        blit_sprite(output_grid, scaled_shape, x=scaled_shape.shape[1], y=scaled_shape.shape[0], background=Color.BLACK)  # Bottom-right

    return output_grid

# Constructing the color map
color_map = {
    Color.RED: Color.BLUE,
    Color.BLUE: Color.GREEN,
    Color.GREEN: Color.YELLOW,
    Color.YELLOW: Color.RED,
    Color.PINK: Color.PURPLE,
    Color.PURPLE: Color.BROWN,
    Color.BROWN: Color.GRAY,
    Color.GRAY: Color.PINK,
    Color.ORANGE: Color.ORANGE,
    Color.BLACK: Color.BLACK
}