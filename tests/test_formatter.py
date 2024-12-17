import numpy as np
from arc.core import Example, Task
from arc.tokenizers import Formatter


def get_formatter() -> Formatter:
    return Formatter(
        preprompt="hi",
        output_tail_token="<eos>",
        input_head_token="I",
        output_head_token="O",
    )


def test_format_grid():
    f = get_formatter()
    grid = np.array([[0, 1], [2, 3]])

    assert f.format_grid(grid) == "01\n23\n"


def test_format_task():
    f = get_formatter()
    train_input_grid = np.array([[0]])
    train_output_grid = np.array([[1]])
    test_input_grid = np.array([[3]])
    test_output_grid = np.array([[4]])
    task = Task(
        train_set=[Example(train_input_grid, train_output_grid)],
        test=Example(test_input_grid, test_output_grid),
    )
    formatter = Formatter(
        preprompt="hi",
        output_tail_token="<eos>",
        input_head_token="I",
        output_head_token="O",
    )
    assert (
        formatter.format_task(task, include_test_output=True)
        == "hiI0\nO1\n<eos>I3\nO4\n<eos>"
    )


def test_format_task_no_test_output():
    f = get_formatter()
    train_input_grid = np.array([[0]])
    train_output_grid = np.array([[1]])
    test_input_grid = np.array([[3]])
    test_output_grid = np.array([[4]])
    task = Task(
        train_set=[Example(train_input_grid, train_output_grid)],
        test=Example(test_input_grid, test_output_grid),
    )
    formatter = Formatter(
        preprompt="hi",
        output_tail_token="<eos>",
        input_head_token="I",
        output_head_token="O",
    )
    assert (
        formatter.format_task(task, include_test_output=False) == "hiI0\nO1\n<eos>I3\nO"
    )
