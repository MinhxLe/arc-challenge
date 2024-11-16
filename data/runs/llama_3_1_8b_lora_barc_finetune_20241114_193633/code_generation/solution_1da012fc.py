from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, symmetry, reflection

# description:
# In the input, you will see a grid filled with various colored pixels. The goal is to create a new output grid where each colored pixel is reflected across the center of the grid. If a pixel is located at (x, y), its reflection will be at (n-1-x, m-1-y) where n and m are the dimensions of the grid. Pixels that are black should remain black in the output grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    n, m = input_grid.shape

    for x in range(n):
        for y in range(m):
            if input_grid[x, y]!= Color.BLACK:
                reflected_x = n - 1 - x
                reflected_y = m - 1 - y
                output_grid[reflected_x, reflected_y] = input_grid[x, y]

    return output_grid