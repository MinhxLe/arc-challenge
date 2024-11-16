from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pixel shifting, color blending, averaging

# description:
# In the input, you will see a grid filled with colored pixels. To create the output, shift all the colored pixels one position to the right and fill in the empty spaces with the average color of the shifted pixels.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to hold the output
    output_grid = np.copy(input_grid)
    
    # Get the dimensions of the grid
    rows, cols = input_grid.shape
    
    # Shift all colored pixels one position to the right
    for x in range(rows):
        for y in range(cols):
            if input_grid[x, y]!= Color.BLACK:  # Only shift colored pixels
                # Calculate the new position
                new_y = y + 1
                
                # If the new position is out of bounds, we will set it to the average color of the original pixel
                if new_y >= cols:
                    # Calculate the average color of the shifted pixel
                    color_count = np.bincount(input_grid[x, y].flatten())
                    average_color = Color.BLACK if np.all(color_count == 0) else color_count[np.argmax(color_count)]
                    output_grid[x, new_y - 1] = average_color
                
    return output_grid