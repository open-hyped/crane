from datasets import Dataset, DatasetDict, load_from_disk

from crane.arrow import ArrowDatasetWriter

from .base import BaseTestDatasetWriter


class TestArrowDatasetWriter(BaseTestDatasetWriter):
    dataset = Dataset.from_dict({"obj": list(range(10))})
    writer_type = ArrowDatasetWriter

    def execute_test(self) -> None:
        # load dataset from disk
        actual_ds = load_from_disk(".")
        # compare to source dataset
        for actual, expected in zip(actual_ds, type(self).dataset):
            assert actual == expected


class TestArrowDatasetWriter_DatasetDict(BaseTestDatasetWriter):
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"obj": list(range(0, 10))}),
            "test": Dataset.from_dict({"obj": list(range(10, 20))}),
        }
    )
    writer_type = ArrowDatasetWriter

    def execute_test(self) -> None:
        # load dataset from disk
        actual_ds = load_from_disk(".")

        for split, ds in type(self).dataset.items():
            assert split in actual_ds
            # compare to source dataset
            for actual, expected in zip(actual_ds[split], ds):
                assert actual == expected
