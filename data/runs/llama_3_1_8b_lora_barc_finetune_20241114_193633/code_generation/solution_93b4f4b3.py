from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# object detection, resizing, color mapping

# description:
# In the input, you will see several colored objects on a black background. 
# To create the output, resize each object to fit within a bounding box defined by its original shape, 
# and change its color to the average color of its pixels (ignoring black).

def transform(input_grid):
    # Step 1: Detect all objects in the input grid
    objects = detect_objects(grid=input_grid, background=Color.BLACK, monochromatic=False, connectivity=4)
    
    # Step 2: Create an output grid to hold the resized objects
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Step 3: For each detected object, resize it and change its color
    for obj in objects:
        # Crop the object to get the actual shape
        cropped_object = crop(obj, background=Color.BLACK)
        
        # Calculate the average color (ignoring black)
        color_counts = {}
        for color in cropped_object.flatten():
            if color!= Color.BLACK:
                if color not in color_counts:
                    color_counts[color] = 0
                color_counts[color] += 1
        
        # Find the most frequent color (in case of ties, we can choose the first one)
        new_color = max(color_counts, key=color_counts.get)
        
        # Get the bounding box of the cropped object
        x, y, width, height = bounding_box(cropped_object, background=Color.BLACK)

        # Create a new grid for the resized object
        resized_object = np.full((width, height), Color.BLACK)
        resized_object[:] = new_color
        
        # Resize the object to fit in the output grid
        blit_sprite(output_grid, resized_object, x=x, y=y, background=Color.BLACK)

    return output_grid