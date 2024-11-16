from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# fractals, recursive patterns, color mapping

# description:
# In the input, you will see a simple fractal pattern in the center of the grid.
# To create the output, recursively replicate the fractal pattern across the grid, 
# maintaining the original color scheme. Each recursive iteration should be represented 
# in a smaller version of the original fractal pattern.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the bounding box of the fractal pattern
    objects = find_connected_components(input_grid, monochromatic=False, connectivity=4)
    
    # Assuming there is one main object, we will work with it
    fractal_object = objects[0]
    bounding_box_fractal = bounding_box(fractal_object)
    
    # Crop the fractal pattern
    fractal_pattern = crop(fractal_object, background=Color.BLACK)
    
    # Determine the size of the output grid
    output_size = max(input_grid.shape) * 3  # Each side will be tripled
    output_grid = np.full((output_size, output_size), Color.BLACK)

    # Center the fractal pattern in the output grid
    center_x = output_size // 2
    center_y = output_size // 2
    pattern_x = bounding_box_fractal[0]
    pattern_y = bounding_box_fractal[1]
    
    # Scale factor for the recursive pattern
    scale_factor = 1
    
    while scale_factor <= 3:  # Three iterations
        # Calculate the size of the current fractal
        current_size = (fractal_pattern.shape[0] * scale_factor, fractal_pattern.shape[1] * scale_factor)
        
        # Calculate the position to blit the fractal pattern
        start_x = center_x - current_size[0] // 2
        start_y = center_y - current_size[1] // 2
        
        # Blit the scaled fractal pattern onto the output grid
        blit_sprite(output_grid, fractal_pattern, x=start_x, y=start_y, background=Color.BLACK)
        
        # Scale the pattern for the next iteration
        fractal_pattern = np.repeat(fractal_pattern, 2, axis=0)  # Scale vertically
        fractal_pattern = np.repeat(fractal_pattern, 2, axis=1)  # Scale horizontally
        
        scale_factor += 1

    return output_grid