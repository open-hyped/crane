import glob
import os

import pyarrow.dataset as pds
import pyarrow.parquet as pq
import pytest
from datasets import Dataset, DatasetDict, load_dataset

from crane import ParquetDatasetWriter

from .base import BaseTestDatasetWriter


class TestParquetDatasetWriter(BaseTestDatasetWriter):
    dataset = Dataset.from_dict({"obj": list(range(10))})
    writer_type = ParquetDatasetWriter

    def execute_test(self) -> None:
        # load dataset from disk
        actual_ds = load_dataset("parquet", data_files="shard-*.parquet", split="train")
        # compare to source dataset
        for actual, expected in zip(actual_ds, type(self).dataset):
            assert actual == expected


class TestParquetDatasetWriter_DatasetDict(BaseTestDatasetWriter):
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"obj": list(range(0, 10))}),
            "test": Dataset.from_dict({"obj": list(range(10, 20))}),
        }
    )
    writer_type = ParquetDatasetWriter

    def execute_test(self) -> None:
        for split, ds in type(self).dataset.items():
            actual_ds = load_dataset(
                "parquet", data_files=f"{split}/shard-*.parquet", split="train"
            )
            # compare to source dataset
            for actual, expected in zip(actual_ds, ds):
                assert actual == expected


class TestParquetDatasetWriter_Compression(BaseTestDatasetWriter):
    dataset = Dataset.from_dict({"obj": list(range(100)), "text": ["a text"] * 100})
    writer_type = ParquetDatasetWriter
    writer_args = {"compression": "zstd"}

    def execute_test(self) -> None:
        # the codec has to reach the file, not just the constructor
        metadata = pq.ParquetFile("shard-00000.parquet").metadata
        codecs = {
            metadata.row_group(group).column(column).compression
            for group in range(metadata.num_row_groups)
            for column in range(metadata.num_columns)
        }
        assert codecs == {"ZSTD"}

        actual_ds = load_dataset("parquet", data_files="shard-*.parquet", split="train")
        for actual, expected in zip(actual_ds, type(self).dataset):
            assert actual == expected


class TestParquetDatasetWriter_RowGroupSize(BaseTestDatasetWriter):
    dataset = Dataset.from_dict({"obj": list(range(100))})
    writer_type = ParquetDatasetWriter
    # smaller than the batches the writer is handed, so it has to split them
    writer_args = {"row_group_size": 10, "write_batch_size": 50}

    def execute_test(self) -> None:
        metadata = pq.ParquetFile("shard-00000.parquet").metadata
        assert metadata.num_rows == 100
        assert metadata.num_row_groups == 10


class TestParquetDatasetWriter_Metadata(BaseTestDatasetWriter):
    dataset = Dataset.from_dict({"obj": list(range(100))})
    writer_type = ParquetDatasetWriter
    # Small shards, so there is more than one for `_metadata` to summarise. `write_batch_size`
    # is what makes that possible: a shard rolls over on what has reached the file, and
    # nothing reaches it until a batch is full.
    writer_args = {"max_shard_size": 1024, "write_batch_size": 10}

    def execute_test(self) -> None:
        shards = sorted(glob.glob("shard-*.parquet"))
        assert len(shards) > 1, "the test needs several shards to be meaningful"

        metadata = pq.read_metadata("_metadata")
        # it describes the whole dataset, not one shard
        assert metadata.num_rows == sum(pq.read_metadata(s).num_rows for s in shards)
        assert metadata.num_rows == len(type(self).dataset)

        # every shard is named, by the path a reader resolves relative to this directory
        named = {metadata.row_group(i).column(0).file_path for i in range(metadata.num_row_groups)}
        assert named == set(shards)

        # the info file crane writes for every writer is still there and untouched
        assert os.path.exists("dataset_info.json")
        assert os.path.exists("state.json")

        # a reader planning from `_metadata` alone sees the whole dataset - this is the
        # point of the file. `pds.parquet_dataset` is the API that takes one; handing
        # `_metadata` to a reader expecting data files fails, since it holds only footers.
        from_metadata = pds.parquet_dataset("_metadata").to_table()
        assert from_metadata.num_rows == len(type(self).dataset)
        assert from_metadata.column("obj").to_pylist() == list(range(100))

        # and the data still reads back whole
        actual_ds = load_dataset("parquet", data_files="shard-*.parquet", split="train")
        for actual, expected in zip(actual_ds, type(self).dataset):
            assert actual == expected


class TestParquetDatasetWriter_MetadataDisabled(BaseTestDatasetWriter):
    dataset = Dataset.from_dict({"obj": list(range(10))})
    writer_type = ParquetDatasetWriter
    writer_args = {"write_metadata": False}

    def execute_test(self) -> None:
        assert not os.path.exists("_metadata")
        # the rest of the output is unaffected
        assert os.path.exists("dataset_info.json")
        assert glob.glob("shard-*.parquet")


class TestParquetDatasetWriter_MetadataForDatasetDict(BaseTestDatasetWriter):
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"obj": list(range(0, 10))}),
            "test": Dataset.from_dict({"obj": list(range(10, 20))}),
        }
    )
    writer_type = ParquetDatasetWriter

    def execute_test(self) -> None:
        # one `_metadata` per split, each describing only its own shards
        for split, ds in type(self).dataset.items():
            metadata = pq.read_metadata(os.path.join(split, "_metadata"))
            assert metadata.num_rows == len(ds)


class TestRowGroupSize:
    """Row groups are the unit a reader skips and the unit compression works over.

    `ParquetWriter.write_table` emits at least one row group per call, so writing pipeline
    batches straight through gave a row group per batch - 4000 of them for 4000 rows when
    the pipeline delivered one row at a time.
    """

    @staticmethod
    def _row_groups(save_dir: str) -> int:
        shard = next(f for f in os.listdir(save_dir) if f.endswith(".parquet"))
        return pq.ParquetFile(os.path.join(save_dir, shard)).metadata.num_row_groups

    @pytest.fixture
    def ds(self):
        return Dataset.from_dict({"x": list(range(1000))}).to_iterable_dataset(num_shards=1)

    @pytest.mark.parametrize("prefetch", [1, 8, 256])
    def test_the_pipeline_batch_size_does_not_decide_the_file_layout(self, ds, tmp_path, prefetch):
        out = str(tmp_path / f"out-{prefetch}")
        ParquetDatasetWriter(
            out,
            write_batch_size=500,
            prefetch_factor=prefetch,
            num_proc=1,
            disable_tqdm=True,
            write_metadata=False,
        ).write(ds)

        assert self._row_groups(out) == 2

    def test_every_row_survives_being_buffered(self, ds, tmp_path):
        out = str(tmp_path / "out")
        ParquetDatasetWriter(
            out, write_batch_size=512, num_proc=1, disable_tqdm=True, write_metadata=False
        ).write(ds)

        loaded = load_dataset("parquet", data_files=os.path.join(out, "*.parquet"))["train"]
        assert sorted(loaded["x"]) == list(range(1000))
