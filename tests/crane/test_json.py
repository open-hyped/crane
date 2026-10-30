import json
import os

import pyarrow as pa
from datasets import IterableDataset
from datasets.iterable_dataset import ArrowExamplesIterable

from crane import JsonDatasetWriter

from .base import BaseTestDatasetWriter


class BaseTestJsonDatasetWriter(BaseTestDatasetWriter):
    writer_type = JsonDatasetWriter

    def execute_test(self) -> None:
        cls = type(self)

        assert "shard-00000.json" in os.listdir(".")
        with open("shard-00000.json", "r") as f:
            output_samples = list(map(json.loads, f.readlines()))

        for actual, expected in zip(output_samples, cls.dataset.with_format(), strict=True):
            assert actual == expected


class TestJsonDatasetWriterPythonFormat(BaseTestJsonDatasetWriter):
    dataset = IterableDataset.from_generator(
        lambda: iter([{"obj": i} for i in range(10)])
    ).with_format(type="python")


class TestJsonDatasetWriterArrowFormat(BaseTestJsonDatasetWriter):
    dataset = IterableDataset.from_generator(
        lambda: iter([{"obj": i} for i in range(10)])
    ).with_format(type="arrow")


def _uneven_tables(sizes: tuple[int, ...]):
    """Tables of `sizes` rows, the rows numbered in the order they are yielded; a tuple, as
    `datasets` would make a shard of each item of a list."""
    start = 0
    for i, size in enumerate(sizes):
        yield i, pa.table({"obj": list(range(start, start + size))})
        start += size


class TestJsonDatasetWriterKeepsTheOrder(BaseTestJsonDatasetWriter):
    """A source of tables of uneven sizes is written in its order, though it is prefetched in
    batches of 3 and written in batches of 4: rebatching from the one to the other must not let a
    full batch overtake the rows buffered before it, nor the rest of a cut chunk fall behind the
    chunks after it."""

    dataset = IterableDataset(
        ArrowExamplesIterable(_uneven_tables, {"sizes": (3, 8, 5, 10, 1, 4, 0, 9, 2, 7) * 10})
    )
    writer_args = {"prefetch_factor": 3, "write_batch_size": 4}


class TestJsonDatasetWriterKeepsTheOrderArrowFormat(TestJsonDatasetWriterKeepsTheOrder):
    dataset = TestJsonDatasetWriterKeepsTheOrder.dataset.with_format(type="arrow")
