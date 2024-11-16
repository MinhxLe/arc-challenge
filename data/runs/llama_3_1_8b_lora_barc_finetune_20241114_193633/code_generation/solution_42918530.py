from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object alignment, color transformation

# description:
# In the input, you will see several colored objects arranged in a row on a black background. 
# Each object has a specific color and a bounding box around it. The goal is to align all objects to the leftmost side of the grid, 
# while changing their colors based on their original color: 
# - If an object's color is red, change it to blue.
# - If an object's color is green, change it to yellow.
# - If an object's color is blue, change it to orange.
# - If an object's color is yellow, change it to pink.
# - If an object's color is any other color, change it to black.

def transform(input_grid):
    # Plan:
    # 1. Extract the objects from the grid.
    # 2. Change their colors based on the color transformation rules.
    # 3. Align all objects to the leftmost side of the grid.

    # Step 1: Extract objects from the input grid
    objects = detect_objects(grid=input_grid, monochromatic=True, background=Color.BLACK, connectivity=4, allowed_dimensions=[(3, 3), (4, 4), (5, 5)])

    # Step 2: Create a new grid for the output
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Step 3: Transform the colors and align objects to the left
    for obj in objects:
        # Crop the object to get its color
        sprite = crop(obj, background=Color.BLACK)
        
        # Identify the color of the sprite
        original_color = sprite[0, 0]  # Assuming the sprite is a solid color in its bounding box
        
        # Apply color transformation
        new_color = Color.BLACK  # Default to black if not recognized
        if original_color == Color.RED:
            new_color = Color.BLUE
        elif original_color == Color.GREEN:
            new_color = Color.YELLOW
        elif original_color == Color.BLUE:
            new_color = Color.ORANGE
        elif original_color == Color.YELLOW:
            new_color = Color.PINK
        
        # Change the color of the sprite
        sprite[sprite!= Color.BLACK] = new_color

        # Find the bounding box of the sprite
        x, y, width, height = bounding_box(sprite, background=Color.BLACK)
        
        # Place the transformed sprite into the output grid aligned to the leftmost side
        blit_sprite(output_grid, sprite, x=0, y=y, background=Color.BLACK)

    return output_grid