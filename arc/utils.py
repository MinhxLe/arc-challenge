import numpy as np
from arc.core import Color, Grid
from itertools import islice
from typing import Iterator, TypeVar, Iterable, Tuple

T = TypeVar("T")  # Generic type for any item


def ndarray_to_tuple(array: np.ndarray) -> Tuple:
    if array.ndim == 1:
        return tuple(array)
    else:
        return tuple(ndarray_to_tuple(row) for row in array)


def tuple_to_ndarray(tup: Tuple) -> np.ndarray:
    if not isinstance(tup[0], tuple):
        return np.array(tup)
    else:
        return np.array([tuple_to_ndarray(row) for row in tup])


def create_color_array(arr: Grid) -> str:
    """
    Convert a Grid to a string representation of color names.

    Args:
        arr (Grid): 2D array of integers corresponding to Color enum values

    Returns:
        str: Multi-line string with color names separated by spaces
    """

    # Convert each row to color names and join with spaces
    rows = []
    for row in arr:
        color_names = [Color(val).name.capitalize() for val in row]
        rows.append(" ".join(color_names))

    # Join rows with newlines to create final output
    return "\n".join(rows)


def batch(iterable: Iterable[T], size: int) -> Iterator[Tuple[T, ...]]:
    """
    Batch data into tuples of length `size`. The last batch may be shorter.

    Args:
        iterable: Any iterable to batch
        size: Size of each batch

    Returns:
        Iterator yielding tuples of length `size` or less
    """
    iterator = iter(iterable)
    while True:
        batch = tuple(islice(iterator, size))
        if not batch:
            break
        yield batch
