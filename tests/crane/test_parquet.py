import pyarrow.parquet as pq
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
