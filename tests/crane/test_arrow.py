import os

import datasets
import pyarrow as pa
import pytest
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


class TestRecordBatchSize:
    """The file's layout must not be decided by the pipeline's batch size.

    Written straight through, a pipeline batch of one row becomes a record batch of one row.
    Arrow stream files carry no index, so every reader then parses one header per row: a real
    corpus here held 831,232 record batches in a 5 GB shard and took 129s to open.
    """

    @staticmethod
    def _batches(save_dir: str) -> list[int]:
        shard = next(f for f in os.listdir(save_dir) if f.endswith(".arrow"))
        with pa.memory_map(os.path.join(save_dir, shard), "rb") as source:
            return [batch.num_rows for batch in pa.ipc.open_stream(source)]

    @pytest.fixture
    def ds(self):
        return Dataset.from_dict({"x": list(range(1000))}).to_iterable_dataset(num_shards=1)

    @pytest.mark.parametrize("prefetch", [1, 8, 256])
    def test_the_pipeline_batch_size_does_not_decide_the_file_layout(self, ds, tmp_path, prefetch):
        out = str(tmp_path / f"out-{prefetch}")
        ArrowDatasetWriter(
            out,
            write_batch_size=500,
            prefetch_factor=prefetch,
            num_proc=1,
            disable_tqdm=True,
        ).write(ds)

        assert self._batches(out) == [500, 500]

    def test_every_row_survives_being_buffered(self, ds, tmp_path):
        # the failure this guards against loses rows silently: nothing downstream knows a
        # buffered row existed
        out = str(tmp_path / "out")
        ArrowDatasetWriter(out, write_batch_size=512, num_proc=1, disable_tqdm=True).write(ds)

        assert sum(self._batches(out)) == 1000
        assert sorted(datasets.load_from_disk(out)["x"]) == list(range(1000))

    def test_a_remainder_smaller_than_the_threshold_is_still_written(self, ds, tmp_path):
        # 1000 rows at 300 per batch leaves 100 that only `finalize_shard` will write
        out = str(tmp_path / "out")
        ArrowDatasetWriter(out, write_batch_size=300, num_proc=1, disable_tqdm=True).write(ds)

        assert self._batches(out) == [300, 300, 300, 100]

    def test_one_row_per_batch_is_still_available(self, ds, tmp_path):
        # the old behaviour, for a caller who wants it
        out = str(tmp_path / "out")
        ArrowDatasetWriter(
            out,
            write_batch_size=1,
            prefetch_factor=1,
            num_proc=1,
            disable_tqdm=True,
        ).write(ds)

        assert len(self._batches(out)) == 1000
