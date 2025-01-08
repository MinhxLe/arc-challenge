import functools
import traceback
from typing import Callable, TypeVar, ParamSpec
from loguru import logger

# Type variables for maintaining function signature
P = ParamSpec("P")
T = TypeVar("T")


def log_exceptions(msg: str):
    """msg
    Decorator that logs exceptions with a custom msg before reraising.

    Args:
        msg (str): Prefix to add to the log message

    Example:
        @log_exceptions("DataLoader")
        def load_data():
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"{msg}: {str(e)}")
                raise

        return wrapper

    return decorator
