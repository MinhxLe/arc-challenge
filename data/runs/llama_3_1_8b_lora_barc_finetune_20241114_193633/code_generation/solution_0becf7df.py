from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color blending, pixel transformation

# description:
# In the input, you will see a grid containing a colorful object surrounded by a black background.
# To create the output, replace the object with a new color based on the average color of its pixels.
# The output grid should maintain the same dimensions as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid that is a copy of the input grid
    output_grid = np.copy(input_grid)

    # Find all the non-background pixels (colors) in the input grid
    non_background_pixels = np.argwhere(input_grid!= Color.BLACK)

    # Calculate the average color of the non-background pixels
    if non_background_pixels.size > 0:
        # Extract the colors
        colors = input_grid[non_background_pixels[0, :].astype(int)]
        
        # Compute the average color (as a simple average of color indices)
        average_color = np.mean(colors).astype(int)

        # Fill the output grid with the average color where the original object was present
        output_grid[output_grid!= Color.BLACK] = average_color

    return output_grid