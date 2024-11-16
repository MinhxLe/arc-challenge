from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color mapping, boundary extraction

# description:
# In the input, you will see a grid with a colored object surrounded by a black background.
# To make the output, you should extract the boundary of the object and fill it with a new color, while keeping the object's interior intact.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to work on
    output_grid = np.copy(input_grid)

    # Find the connected components (the object) in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False, connectivity=4)

    # Identify the main object (the largest connected component)
    object_components = sorted(objects, key=lambda obj: np.sum(obj!= Color.BLACK), reverse=True)

    # Assuming there is only one object, take the first one
    main_object = object_components[0]

    # Crop the object to get its bounding box
    x, y, width, height = bounding_box(main_object)

    # Get the boundary of the object
    boundary_mask = np.zeros_like(output_grid, dtype=int)
    boundary_mask[x:x + width, y:y + height] = main_object[x:x + width, y:y + height]

    # Create the mask for the interior of the object
    interior_mask = np.zeros_like(output_grid, dtype=int)
    interior_mask[boundary_mask == main_object[0, 0]] = Color.BLACK

    # Fill the boundary with a new color (let's say Color.BLUE for this example)
    boundary_color = Color.BLUE
    boundary_mask[boundary_mask!= Color.BLACK] = boundary_color

    # Replace the interior with black (background)
    output_grid[interior_mask!= Color.BLACK] = Color.BLACK

    # Fill the boundary back into the output grid
    output_grid[boundary_mask!= Color.BLACK] = boundary_color

    return output_grid