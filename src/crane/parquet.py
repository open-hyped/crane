"""Module providing the :class:`ParquetDatasetWriter` class.

The :class:`ParquetDatasetWriter` class writes dataset samples to individual Parquet shard
files, with each worker writing a separate shard.
"""

import glob
import logging
from typing import Any

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import DatasetInfo

from .core import BaseDatasetWriter
from .core.worker import get_worker_info

logger = logging.getLogger(__name__)


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

    A `_metadata` file summarising every shard is written alongside, so a Parquet reader
    can plan a scan of the whole directory without opening each shard first:

    .. code-block:: python

        import pyarrow.dataset as pds

        table = pds.parquet_dataset("./data/_metadata").to_table()

    Note that pointing a reader at the directory itself does not work - it would try to
    parse `dataset_info.json` as Parquet. Use `_metadata`, or a `shard-*.parquet` glob.
    """

    def __init__(
        self,
        save_dir: str,
        compression: str = "snappy",
        row_group_size: None | int = None,
        write_metadata: bool = True,
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
            write_metadata (bool): Whether to write a `_metadata` file summarising every
                shard once the run finishes. Defaults to True; see
                :func:`finalize_dataset` for what it buys and what it costs.
            **kwargs (Any): Forwarded to :class:`BaseDatasetWriter`, which documents the
                sharding, parallelism and progress options shared by every writer.
        """
        super(ParquetDatasetWriter, self).__init__(save_dir, **kwargs)
        self._compression = compression
        self._row_group_size = row_group_size
        self._write_metadata = write_metadata

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

    def finalize_dataset(self, ds: datasets.IterableDataset) -> None:
        """Write a `_metadata` file summarising every shard.

        `_metadata` is a footer-only Parquet file holding each shard's row-group metadata
        with its path attached. A reader handed the directory can plan a scan of the whole
        dataset from that one file, instead of opening every shard to learn the schema,
        the row counts and the row-group statistics - and it can then skip shards using
        statistics it has not had to read. `dataset_info.json` and `state.json` are
        written as for any other crane writer; this is additional, not a replacement.

        The shards are written by different processes, so their metadata cannot simply be
        accumulated as they go. It is gathered here instead, by reading each footer back
        now that they all exist. A footer is the tail of a file rather than its data, so
        this costs one small read per shard, once.

        Args:
            ds (datasets.IterableDataset): The dataset that was written.
        """
        if not self._write_metadata:
            return

        shard_paths = sorted(glob.glob("shard-*.parquet"))
        if not shard_paths:
            # Nothing was written. A `_metadata` here would describe an empty dataset as
            # though that were the intended result; leave it out so the directory looks
            # as empty as it is.
            logger.warning("No shards were written, skipping the parquet metadata file.")
            return

        metadata = []
        for path in shard_paths:
            shard_metadata = pq.read_metadata(path)
            # The path a reader resolves, relative to the directory holding `_metadata`.
            shard_metadata.set_file_path(path)
            metadata.append(shard_metadata)

        pq.write_metadata(metadata[0].schema.to_arrow_schema(), "_metadata", metadata)
        logger.info(f"Wrote parquet metadata for {len(shard_paths)} shards.")
