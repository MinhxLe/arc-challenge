from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel grouping, color transformation

# description:
# In the input, you will see a grid filled with colored pixels on a black background. The colors are arranged in various groups.
# To create the output, transform the grid by grouping the pixels of the same color into larger blocks. 
# Each block will be a solid color and will occupy a rectangular area in the output grid, where the dimensions of each block will depend on the number of pixels of the same color that are grouped together.

def transform(input_grid):
    # Create an output grid filled with black
    output_grid = np.zeros_like(input_grid)

    # Find connected components in the grid
    components = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True)

    # Create a mapping of colors to their respective blocks
    color_blocks = {}
    for component in components:
        color = component[0, 0]  # Get the color of the component
        # Get the bounding box of the component
        x, y, width, height = bounding_box(component, background=Color.BLACK)

        # Create a block in the output grid for the color
        color_blocks[color] = crop(component, background=Color.BLACK)

    # Determine the maximum size of blocks based on the number of unique colors
    max_width = max(len(color_blocks), len(components))
    max_height = max(len(color_blocks))

    # Calculate the positions to place the blocks in the output grid
    for idx, (color, block) in enumerate(color_blocks.items()):
        # Calculate the position of the block in the output grid
        output_x = idx // max_height
        output_y = idx % max_height

        # Place the block in the output grid
        blit_sprite(output_grid, block, x=output_x, y=output_y, background=Color.BLACK)

    return output_grid