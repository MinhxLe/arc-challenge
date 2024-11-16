from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern extraction, rotation, alignment

# description:
# In the input, you will see a grid containing a series of identical patterns arranged in a grid-like formation, 
# with one of them occluded by a black rectangle. To create the output, extract the visible patterns, 
# rotate them into their original positions, and remove the occluding rectangle.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Plan:
    # 1. Extract the visible patterns by finding the connected components
    # 2. Remove the occluding rectangle
    # 3. Rotate the visible patterns back into their original positions
    # 4. Create the output grid and place the rotated patterns

    # Find the occluding rectangle (black rectangle)
    occlusion_color = Color.BLACK
    black_rectangle_mask = (input_grid == occlusion_color)

    # Find the connected components (the visible patterns)
    visible_components = find_connected_components(input_grid, background=occlusion_color, monochromatic=False)

    # Create an output grid initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # For each visible pattern, rotate it back and place it in the output grid
    for component in visible_components:
        # Crop the component to get the sprite
        sprite = crop(component, background=occlusion_color)
        
        # Find the bounding box of the sprite
        x, y, width, height = bounding_box(sprite)
        
        # Rotate the sprite back to its original position
        # Here we assume a simple rotation for demonstration
        # This could be enhanced with more complex rotation logic
        rotated_sprite = np.rot90(sprite, k=-1)  # Rotate counterclockwise 90 degrees

        # Calculate the position to place the rotated sprite in the output grid
        output_x = (x + (input_grid.shape[0] - rotated_sprite.shape[0]) // 2)
        output_y = (y + (input_grid.shape[1] - rotated_sprite.shape[1]) // 2)

        # Place the rotated sprite into the output grid
        blit_sprite(output_grid, rotated_sprite, x=output_x, y=output_y, background=Color.BLACK)

    return output_grid