from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, radial symmetry

# description:
# In the input, you will see a colored object in the center of a grid with a single pixel at the top right corner that indicates the center point.
# To create the output grid, replicate the object in a radial pattern around the center point, duplicating the object's color and shape symmetrically in all directions.

def transform(input_grid):
    # Create an output grid that is the same size as the input grid.
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Find the center point of the object (the pixel in the top right corner).
    center_x = 0
    center_y = 0

    # Get the color of the center pixel
    center_color = input_grid[center_x, center_y]

    # Find the object's bounding box
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4)
    main_object = objects[0]  # Assume there's only one object in the center

    # Get the bounding box of the object
    obj_x, obj_y, obj_width, obj_height = bounding_box(main_object, background=Color.BLACK)

    # Crop the object to get the sprite
    sprite = crop(main_object, background=Color.BLACK)

    # Get the dimensions of the sprite
    sprite_height, sprite_width = sprite.shape

    # Define the number of duplications in all directions
    num_duplicates = 8  # Number of copies in a radial pattern (0 to 8)

    for i in range(num_duplicates):
        for j in range(num_duplicates):
            # Calculate the position for each copy
            copy_x = center_x + i * (sprite_width + 1) - sprite_width // 2
            copy_y = center_y + j * (sprite_height + 1) - sprite_height // 2
            
            # Place the sprite in the output grid
            blit_sprite(output_grid, sprite, x=copy_x, y=copy_y, background=Color.BLACK)

    return output_grid