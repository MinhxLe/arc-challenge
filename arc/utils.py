import numpy as np
from typing import Tuple


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
