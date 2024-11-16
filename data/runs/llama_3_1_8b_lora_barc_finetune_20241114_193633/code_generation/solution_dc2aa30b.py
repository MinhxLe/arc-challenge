from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, pixel mirroring, color transformation

# description:
# In the input, you will see a grid with colored pixels and a black background. The grid will contain
# a symmetric pattern along the vertical axis. To create the output grid, mirror the pixels on one side of the
# vertical axis to the other side. The output grid should maintain the original colors on one side and
# fill the mirrored pixels on the opposite side with the same color.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to the input grid
    output_grid = np.copy(input_grid)

    # Get the number of columns
    cols = input_grid.shape[1]

    # Iterate over the left half of the grid
    for row in range(input_grid.shape[0]):
        for col in range(cols // 2):
            # Get the color of the current pixel
            current_color = input_grid[row, col]
            # Calculate the corresponding mirrored position
            mirrored_col = cols - col - 1
            
            # Fill the mirrored pixel with the current pixel's color
            if current_color!= Color.BLACK:
                output_grid[row, mirrored_col] = current_color

    return output_grid