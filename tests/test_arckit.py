"""
This is not a real test suite. it is mostly to highlight arckit as a library.
"""

import arckit
import arckit.vis as vis
import numpy as np
import pytest

TASK_ID = "4522001f"  # picked arbitrary


def test_read_all_data():
    train_set, test_set = arckit.load_data(version="latest")
    assert len(train_set) == 400
    assert len(test_set) == 400


def test_get_train_from_task():
    task = arckit.load_single(TASK_ID)
    assert isinstance(task.train, list)
    assert isinstance(task.train[0], tuple)
    assert isinstance(task.train[0][0], np.ndarray)


def test_get_test_from_task():
    task = arckit.load_single(TASK_ID)
    assert len(task.test) == 1
    assert isinstance(task.test[0], tuple)
    assert isinstance(task.test[0][0], np.ndarray)


@pytest.mark.skip("not working right now")
def test_draw_task():
    task = arckit.load_single(TASK_ID)
    vis.draw_grid(task.train[0][0], xmax=3, ymax=3)
