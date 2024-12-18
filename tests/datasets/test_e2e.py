from datasets import Dataset
import pytest
from arc.core import Color
from arc.datasets.seed import Datasets
from arc.datasets import transform as t
from arc.transform import Compose, MapColor, Reflect, Rotate


# simple
@pytest.mark.skip()
def test_load_all_datasets():
    print(Datasets.arc_public_train.get_dataset().shape)
    print(Datasets.arc_public_test.get_dataset().shape)
    print(Datasets.concept_arc.get_dataset().shape)
    print(Datasets.arc_heavy.get_dataset().shape)


def test_sample():
    ds = Datasets.arc_public_test.get_dataset()
    ds = t.sample(ds, 10)
    assert len(ds) == 10


def test_concat_dataset():
    ds = t.concat(
        Datasets.arc_public_train.get_dataset(),
        Datasets.arc_public_test.get_dataset(),
    )
    assert isinstance(ds, Dataset)
    assert ds.shape == (800, 2)


def test_shuffle_train_order():
    ds = Datasets.arc_public_test.get_dataset()
    shuffled_ds = t.shuffle_train_order(ds)

    assert any(r1["train"] != r2["train"] for r1, r2 in zip(ds, shuffled_ds))


def test_apply_transform():
    ds = Datasets.arc_public_test.get_dataset()
    transformed_ds = t.apply_transform(ds, Reflect(Reflect.Type.VERTICAL))
    assert any(r1["train"] != r2["train"] for r1, r2 in zip(ds, transformed_ds))
    assert any(r1["test"] != r2["test"] for r1, r2 in zip(ds, transformed_ds))


def test_apply_transform_only_input():
    ds = Datasets.arc_public_test.get_dataset()
    transformed_ds = t.apply_transform(
        ds, Reflect(Reflect.Type.VERTICAL), input_only=True
    )
    assert any(r1["train"] != r2["train"] for r1, r2 in zip(ds, transformed_ds))
    assert any(r1["test"] != r2["test"] for r1, r2 in zip(ds, transformed_ds))

    for r1, r2 in zip(ds, transformed_ds):
        for e1, e2 in zip(r1["train"], r2["train"]):
            assert e1["output"] == e2["output"]
        for e1, e2 in zip(r1["test"], r2["test"]):
            assert e1["output"] == e2["output"]


def test_compose():
    t.concat(
        t.apply_transform(
            t.sample(Datasets.arc_public_test.get_dataset(), 10),
            Compose(
                [
                    Rotate(3),
                    MapColor({Color.RED: Color.BLUE}),
                ]
            ),
        ),
        t.apply_transform(
            t.sample(Datasets.arc_public_test.get_dataset(), 10),
            Compose(
                [
                    Rotate(3),
                    Reflect(Reflect.Type.HORIZONTAL),
                ]
            ).inverse,
        ),
    )
