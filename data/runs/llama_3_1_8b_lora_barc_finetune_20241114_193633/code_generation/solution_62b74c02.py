from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, filling

# description:
# In the input, you will see a grid with a colorful shape that has left-right symmetry.
# To create the output, fill the left half of the shape with a uniform color that matches the color of the right half,
# effectively creating a mirror effect across the vertical center line.

def transform(input_grid):
    # Identify the height and width of the input grid
    height, width = input_grid.shape
    
    # Create an output grid initialized to black
    output_grid = np.full((height, width), Color.BLACK)

    # Fill the output grid with the input grid
    output_grid[:, :width//2] = input_grid[:, :width//2]
    
    # Fill the right half of the output grid with the left half's colors
    output_grid[:, width//2:] = np.flip(input_grid[:, :width//2], axis=1)
    
    return output_grid