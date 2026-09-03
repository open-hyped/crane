"""Module providing the :class:`ParquetDatasetWriter` class.

The :class:`ParquetDatasetWriter` class writes dataset samples to individual Parquet shard
files, with each worker writing a separate shard.
"""

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import DatasetInfo

from .core import BaseDatasetWriter
from .core.worker import get_worker_info


class ParquetDatasetWriter(BaseDatasetWriter):
    """A dataset writer for saving data in Parquet format.

    This class is responsible for writing dataset samples to Parquet files, with each
    worker writing its own shard. The writer converts dataset features to an Arrow schema
    and uses PyArrow for serialization, so no conversion out of Arrow is needed on the
    way to disk.

    Parquet is columnar and compressed, so the output is considerably smaller than the
    Arrow files :class:`ArrowDatasetWriter` produces and reading a subset of columns is
    cheaper. In exchange it is not memory-mappable, which is what
    :func:`datasets.load_from_disk` relies on - load the result as a dataset of parquet
    files instead:

    .. code-block:: python

        writer = ParquetDatasetWriter(save_dir="./data")
        writer.write(ds)

        ds = datasets.load_dataset("parquet", data_dir="./data", split="train")
    """

    def __init__(
        self,
        save_dir: str,
        compression: str = "snappy",
        row_group_size: None | int = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the :class:`ParquetDatasetWriter`.

        Args:
            save_dir (str): Directory where the dataset will be saved.
            compression (str): The codec each column is compressed with, as accepted by
                :class:`pyarrow.parquet.ParquetWriter` - for instance :code:`"snappy"`,
                :code:`"zstd"`, :code:`"gzip"` or :code:`"none"`. Defaults to
                :code:`"snappy"`, which is what PyArrow itself defaults to and trades a
                little compression for being fast to write and read.
            row_group_size (None | int): Maximum number of rows per row group. Row groups
                are the unit a reader can skip, so smaller groups make selective reads
                cheaper and the file slightly larger. Defaults to None, which lets
                PyArrow write one row group per batch handed to it.
            **kwargs (Any): Forwarded to :class:`BaseDatasetWriter`, which documents the
                sharding, parallelism and progress options shared by every writer.
        """
        super(ParquetDatasetWriter, self).__init__(save_dir, **kwargs)
        self._compression = compression
        self._row_group_size = row_group_size

    def initialize_shard(self, shard_id: int, info: DatasetInfo) -> None:
        """Initialize a new Parquet writer shard.

        This method sets up the Parquet file writer for the current shard by:
        - Creating a shard file named :code:`shard-<shard_id>.parquet` for the current worker.
        - Converting dataset features to an Arrow schema.
        - Initializing the Parquet writer to write into that file.

        The working directory is set to the save directory during the write process.

        Args:
            shard_id (int): The id of the shard being initialized.
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        worker_info = get_worker_info()
        # open shard file
        worker_info.ctx.file_path = f"shard-{shard_id:05}.parquet"
        # Buffered, unlike the Arrow writer's raw handle: Parquet writes a footer of
        # metadata on close and builds row groups in memory anyway, so unbuffered writes
        # would only cost syscalls without making anything visible sooner.
        worker_info.ctx.file = open(worker_info.ctx.file_path, "wb")
        # build arrow schema from dataset features
        assert info.features is not None
        worker_info.ctx.schema = info.features.arrow_schema
        # create parquet writer
        worker_info.ctx.writer = pq.ParquetWriter(
            where=worker_info.ctx.file,
            schema=worker_info.ctx.schema,
            compression=self._compression,
        )

    def write_batch_arrow(self, batch: pa.Table) -> int:
        """Write a batch of samples to the Parquet shard.

        Args:
            batch (pa.Table): A batch of samples to be written to the Parquet file.

        Returns:
            int: The number of bytes written to the shard. Parquet compresses and buffers
            row groups, so this reflects what actually reached the file rather than the
            size of the batch, and it can be zero for a batch that is still buffered.
        """
        info = get_worker_info()
        file_size = info.ctx.file.tell()
        # write sample to file
        info.ctx.writer.write_table(
            batch.cast(info.ctx.schema), row_group_size=self._row_group_size
        )
        # return bytes written to file
        return info.ctx.file.tell() - file_size

    def finalize_shard(self, info: DatasetInfo) -> None:
        """Finalize the writing process.

        Closes the Parquet writer, which appends the file's footer, and then the shard
        file itself. The footer is what makes a Parquet file readable at all, so unlike an
        Arrow stream - readable up to wherever it stopped - a shard is only worth anything
        once this has run. The sharding controller pairs every `initialize_shard` with a
        `finalize_shard`, on rollover and at the end of the run alike, so that holds.

        Args:
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        # close writer and file
        worker_info = get_worker_info()
        worker_info.ctx.writer.close()
        worker_info.ctx.file.close()
