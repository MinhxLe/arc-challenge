import numpy as np
from arc.core import Color, Grid
from itertools import islice
from typing import Iterator, TypeVar, Iterable, Tuple, Type, Union
from functools import wraps
import asyncio
from loguru import logger

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


def batch(iterable: Iterable[T], size: int) -> Iterator[list[T, ...]]:
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
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
):
    """
    A decorator that retries an async function if it raises specified exceptions.

    Args:
        max_attempts (int): Maximum number of retry attempts (default: 3)
        delay (float): Initial delay between retries in seconds (default: 1.0)
        backoff_factor (float): Multiplier for delay between retries (default: 2.0)
        exceptions: Exception or tuple of exceptions to catch (default: Exception)

    Returns:
        The decorator function
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}. "
                            f"Retrying in {current_delay:.2f} seconds..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}. "
                            f"Final exception: {str(last_exception)}"
                        )
            assert last_exception is not None
            # If we've exhausted all retries, raise the last exception
            raise last_exception

        return wrapper

    return decorator
