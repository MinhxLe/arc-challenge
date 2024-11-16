from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, cropping, background removal

# description:
# In the input, you will see a grid with several colored objects surrounded by a black background. 
# To create the output, extract the largest object (the largest connected component) and crop it, 
# removing the black background and leaving only the extracted object.

def transform(input_grid):
    # Step 1: Detect all objects in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    
    # Step 2: Identify the largest object
    largest_object = max(objects, key=lambda obj: obj.size)  # largest object by pixel count

    # Step 3: Crop the largest object, removing the background
    output_grid = crop(largest_object, background=Color.BLACK)

    return output_grid