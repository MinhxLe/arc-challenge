import os
from contextlib import contextmanager


@contextmanager
def working_dir(path):
    current_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(current_dir)
