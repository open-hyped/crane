import json
import os

from datasets import IterableDataset

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
