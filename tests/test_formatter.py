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


def test_grid():
    f = get_formatter()
    grid = np.array([[0, 1], [2, 3]])
    serialized_grid = f.format_grid(grid)
    parsed_grid = f.parse_grid(serialized_grid)
    assert serialized_grid == "01\n23\n"
    assert np.all(grid == parsed_grid)


def test_format_task():
    f = Formatter(
        preprompt="hi",
        output_tail_token="<eos>",
        input_head_token="I",
        output_head_token="O",
    )
    train_input_grid = np.array([[0]])
    train_output_grid = np.array([[1]])
    test_input_grid = np.array([[3]])
    test_output_grid = np.array([[4]])
    task = Task(
        id=None,
        train_set=[Example(train_input_grid, train_output_grid)],
        test_set=[Example(test_input_grid, test_output_grid)],
    )
    serialized_task = f.format_task(task, include_test_output=True)
    assert serialized_task == "hiI0\nO1\n<eos>I3\nO4\n<eos>"

    parsed_task = f.parse_task(serialized_task)
    assert parsed_task == task


def test_format_task_no_test_output():
    train_input_grid = np.array([[0]])
    train_output_grid = np.array([[1]])
    test_input_grid = np.array([[3]])
    test_output_grid = np.array([[4]])
    task = Task(
        id=None,
        train_set=[Example(train_input_grid, train_output_grid)],
        test_set=[Example(test_input_grid, test_output_grid)],
    )
    f = Formatter(
        preprompt="hi",
        output_tail_token="<eos>",
        input_head_token="I",
        output_head_token="O",
    )
    assert f.format_task(task, include_test_output=False) == "hiI0\nO1\n<eos>I3\nO"


def test_parse_output_grid():
    train_input_grid = np.array([[0]])
    train_output_grid = np.array([[1]])
    test_input_grid = np.array([[3]])
    test_output_grid = np.array([[4]])
    task = Task(
        id=None,
        train_set=[Example(train_input_grid, train_output_grid)],
        test_set=[Example(test_input_grid, test_output_grid)],
    )
    f = Formatter(
        preprompt="hi",
        output_tail_token="<eos>",
        input_head_token="I",
        output_head_token="O",
    )
    serialized_task = f.format_task(task, include_test_output=True)
    assert np.all(f.parse_test_output_grid(serialized_task) == test_output_grid)
