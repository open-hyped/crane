import glob

import pyarrow as pa
from datasets import Dataset, DatasetDict, load_from_disk

from crane import ArrowDatasetWriter

from .base import BaseTestDatasetWriter


class TestArrowDatasetWriter(BaseTestDatasetWriter):
    dataset = Dataset.from_dict({"obj": list(range(10))})
    writer_type = ArrowDatasetWriter

    def execute_test(self) -> None:
        # load dataset from disk
        actual_ds = load_from_disk(".")
        # compare to source dataset
        self.assert_same_samples(actual_ds, type(self).dataset)


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
            self.assert_same_samples(actual_ds[split], ds)


class TestArrowDatasetWriter_NoEmptyShards(BaseTestDatasetWriter):
    # a single source shard, so with two processes there is nothing for the second one to
    # do - it must not leave a shard behind for the data that never reached it
    dataset = Dataset.from_dict({"obj": list(range(100))}).to_iterable_dataset(1)
    writer_type = ArrowDatasetWriter
    supported_num_procs = (2,)

    def execute_test(self) -> None:
        shards = sorted(glob.glob("shard-*.arrow"))
        assert shards, "the test needs at least one shard to be meaningful"

        for shard in shards:
            with pa.ipc.open_stream(pa.memory_map(shard, "rb")) as reader:
                assert reader.read_all().num_rows > 0, f"{shard} was written without any rows"
