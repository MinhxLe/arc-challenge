from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color filling, symmetry detection

# description:
# In the input, you will see a grid with a central pattern of colored pixels surrounded by a border of black pixels. 
# The task is to fill in the black border with the color of the central pattern, creating a symmetrical shape. 
# The output should be a grid where the border pixels are filled with the color of the central pattern.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to modify for output
    output_grid = np.copy(input_grid)

    # Detect the color of the central pattern
    central_pattern_color = np.unique(output_grid[1:-1, 1:-1])[0]  # Get the color of the central pattern

    # Fill the border with the color of the central pattern
    output_grid[0, :] = central_pattern_color  # Top border
    output_grid[-1, :] = central_pattern_color  # Bottom border
    output_grid[:, 0] = central_pattern_color  # Left border
    output_grid[:, -1] = central_pattern_color  # Right border

    return output_grid