"""Json Dataset Writer.

This module provides a dataset writer implementation for writing samples to a JSON file.
The dataset is sharded, with each worker writing to a separate JSON file. Each sample is
serialized using :code:`orjson` for efficient performance.
"""
import orjson
import pyarrow as pa
import pyarrow.json
from datasets import DatasetInfo
from datasets.iterable_dataset import _batch_to_examples

from .core import BaseDatasetWriter
from .core.worker import get_worker_info


class JsonDatasetWriter(BaseDatasetWriter):
    """A dataset writer for saving data in JSON format.

    This class inherits from :code:`BaseDatasetWriter` and implements methods for writing
    samples in a sharded manner. Each worker writes to a separate JSON file, named
    according to its rank.
    """

    def initialize_shard(self, shard_id: int, info: DatasetInfo) -> None:
        """Initialize the file for writing.

        This method is called at the start of the dataset writing process and
        creates a new JSON file specific to the worker's rank. The file is opened
        in write-binary mode to store serialized JSON lines.

        The file is named using the worker's rank, in the format "shard-<shard_id>.json".
        The working directory is set to the save directory during this method.

        Args:
            shard_id (int): The id of the shard being initialized.
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        info = get_worker_info()
        info.ctx.file_path = f"shard-{shard_id:05}.json"
        info.ctx.file = open(info.ctx.file_path, "wb", buffering=0)

    def write_batch_arrow(self, batch: pa.Table) -> int:
        """Write a batch of samples to the JSON file.

        This method serializes the samples to JSON format using :code:`orjson` and writes
        each sample as a line in the JSON file, with samples separated by newlines. The
        input :code:`batch` is expected to be in the PyArrow :code:`pa.Table` format.

        Args:
            batch (pa.Table): A batch of samples to be written, where each sample is
                represented as a row in the PyArrow table.

        Returns:
            int: The number of bytes written to the shard.
        """
        info = get_worker_info()
        file_size = info.ctx.file.tell()
        # write the sample to the file and return the written bytes
        serialized = b"\n".join(map(orjson.dumps, batch.to_pylist()))
        info.ctx.file.write(serialized + b"\n")
        return info.ctx.file.tell() - file_size

    def write_batch_py(self, batch: pa.Table) -> int:
        """Write a batch of samples format to the JSON file.

        This method serializes the samples to JSON format using :code:`orjson` and writes
        each sample as a line in the JSON file, with samples separated by newlines. The
        input :code:`batch` is expected to be a dictionary where keys are column names
        and values are lists of column data.

        Args:
            batch (dict[str, list[Any]]): A batch of samples to be written, represented as a
                Python dictionary with column names as keys and lists of data as values.

        Returns:
            int: The number of bytes written to the shard.
        """
        info = get_worker_info()
        file_size = info.ctx.file.tell()
        # write the sample to the file and return the written bytes
        serialized = b"\n".join(map(orjson.dumps, _batch_to_examples(batch)))
        info.ctx.file.write(serialized + b"\n")
        return info.ctx.file.tell() - file_size

    def finalize_shard(self, info: DatasetInfo) -> None:
        """Finalize the writing process.

        This method closes the JSON file after all samples have been written.
        The working directory is set to the save directory during this method.

        Args:
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        info = get_worker_info()
        info.ctx.file.close()
