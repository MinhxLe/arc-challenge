from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color mapping

# description:
# In the input, you will see a grid with a colorful sprite and a black background. The sprite has a defined color and is surrounded by black pixels. 
# To create the output, you should reflect the sprite across the vertical and horizontal axes, creating a symmetric pattern. 
# The output grid should be the same size as the input grid, with the original sprite in the center and the reflections surrounding it.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Detect the sprite in the input grid
    sprite_objects = detect_objects(grid=input_grid, colors=Color.NOT_BLACK, monochromatic=False, connectivity=4, allowed_dimensions=None, background=Color.BLACK)

    # Assuming there's only one sprite in the input
    assert len(sprite_objects) == 1
    sprite = sprite_objects[0]

    # Get the bounding box of the sprite
    x, y, width, height = bounding_box(sprite)

    # Crop the sprite to remove any excess black pixels
    cropped_sprite = crop(sprite, background=Color.BLACK)

    # Determine the center of the sprite
    center_x, center_y = x + width // 2, y + height // 2

    # Reflect the sprite across the vertical and horizontal axes
    reflected_sprite_vertical = np.flipud(cropped_sprite)  # Reflect vertically
    reflected_sprite_horizontal = np.fliplr(cropped_sprite)  # Reflect horizontally
    reflected_sprite_both = np.flipud(np.fliplr(cropped_sprite))  # Reflect both ways

    # Place the original sprite in the center of the output grid
    output_grid[center_y:center_y + height, center_x:center_x + width] = cropped_sprite

    # Place the reflected sprite in their respective quadrants
    output_grid[center_y:center_y + height, :width] = reflected_sprite_vertical
    output_grid[:height, center_x:center_x + width] = reflected_sprite_horizontal
    output_grid[:height, :width] = reflected_sprite_both

    return output_grid