import os
from contextlib import contextmanager
import shutil


@contextmanager
def working_dir(path):
    current_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(current_dir)


def rm_all(path: str):
    shutil.rmtree(path)
