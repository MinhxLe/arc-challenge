from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape mirroring, color mapping

# description:
# In the input, you will see a grid containing a colored shape. 
# To create the output, mirror the shape across the vertical axis of the grid and 
# color the mirrored part with a different color.

def transform(input_grid):
    # Detect the colored shape in the input grid
    shape_components = detect_objects(input_grid, colors=[Color.RED], monochromatic=False, connectivity=4)
    
    # Initialize the output grid with the same height and width as input, filled with black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Iterate over each detected shape
    for shape in shape_components:
        # Get the bounding box of the shape
        x, y, w, h = bounding_box(shape)

        # Crop the shape to isolate it
        cropped_shape = crop(shape, background=Color.BLACK)

        # Mirror the shape across the vertical axis
        mirrored_shape = np.flip(cropped_shape, axis=1)

        # Determine the color of the original shape
        original_color = cropped_shape[0, 0]  # Assuming the shape is monochromatic

        # Create a new color for the mirrored shape (different from the original)
        mirrored_color = Color.RED if original_color!= Color.RED else Color.GREEN  # Example mapping

        # Place the original shape in the output grid
        blit_sprite(output_grid, cropped_shape, x=x, y=y, background=Color.BLACK)

        # Place the mirrored shape in the output grid
        blit_sprite(output_grid, mirrored_shape, x=x + w, y=y, background=Color.BLACK, color=mirrored_color)

    return output_grid