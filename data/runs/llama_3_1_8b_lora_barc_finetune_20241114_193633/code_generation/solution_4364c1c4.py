from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object stacking, layering, color separation

# description:
# In the input, you will see a grid with colored objects stacked on top of each other. 
# The output should separate these objects into distinct layers based on their colors. 
# Each layer will be represented by a different color in the output grid.

def transform(input_grid):
    # Create an output grid initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Detect connected components (objects) in the input grid
    objects = detect_objects(input_grid, background=Color.BLACK, connectivity=8, monochromatic=False)

    # Initialize a dictionary to hold the colors and their corresponding layers
    layers = {}
    
    for obj in objects:
        # Get the bounding box of the current object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)
        color = obj[x, y]  # Get the color of the object
        
        if color not in layers:
            layers[color] = []
        
        # Extract the object and place it in the corresponding color layer
        for i in range(height):
            for j in range(width):
                if obj[i, j]!= Color.BLACK:  # Only consider non-background pixels
                    # Place the pixel in the output grid
                    output_grid[y + i, x + j] = color

    return output_grid