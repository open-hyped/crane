import json
import os

from datasets import Dataset

from crane import JsonDatasetWriter

from .base import BaseTestDatasetWriter


class TestJsonDatasetWriter(BaseTestDatasetWriter):
    dataset = Dataset.from_dict({"obj": list(range(10))})
    writer_type = JsonDatasetWriter

    def execute_test(self) -> None:
        cls = type(self)

        assert "shard-00000.json" in os.listdir(".")
        with open("shard-00000.json", "r") as f:
            output_samples = list(map(json.loads, f.readlines()))
            assert len(output_samples) == len(cls.dataset)

        for actual, expected in zip(output_samples, cls.dataset):
            assert actual == expected
