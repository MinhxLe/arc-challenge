from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# merging, symmetry

# description:
# In the input, you will see two distinct colored patterns, each occupying half of a grid.
# To create the output, merge the two patterns by reflecting the first pattern across the center line,
# creating a symmetrical design that combines both halves.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Determine the dimensions of the input grid
    n, m = input_grid.shape
    
    # Create an output grid that is double the height of the input grid
    output_grid = np.full((n, m), Color.BLACK)

    # Split the input grid into two halves
    first_half = input_grid[:n//2, :]
    second_half = input_grid[n//2:, :]

    # Reflect the first half across the center line
    reflected_half = first_half[::-1, :]

    # Place the original first half and the reflected half into the output grid
    output_grid[:n//2, :] = first_half
    output_grid[n//2:, :] = reflected_half

    return output_grid