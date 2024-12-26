"""Module providing the :class:`ArrowDatasetWriter` class.

The :class:`ArrowDatasetWriter` class writes dataset samples to individual Arrow shard files,
with each worker writing a separate shard.
"""

import pyarrow as pa
from datasets import DatasetInfo

from .core import BaseDatasetWriter
from .core.worker import get_worker_info


class ArrowDatasetWriter(BaseDatasetWriter):
    """A dataset writer for saving data in Arrow format.

    This class is responsible for writing dataset samples to Arrow files,
    with each worker writing its own shard. The writer converts dataset
    features to an Arrow schema and uses PyArrow for efficient serialization
    of the data.

    The output is compatible with the Hugging Face Datasets library's :func:`load_from_disk`
    function, making it easy to reload the saved dataset:

    .. code-block:: python

        writer = ArrowDatasetWriter(save_dir="./data")
        writer.write(ds)

        ds = datasets.load_from_disk("./data")
    """

    def initialize_shard(self, shard_id: int, info: DatasetInfo) -> None:
        """Initialize a new Arrow writer shard.

        This method sets up the Arrow file writer for the current shard by:
        - Creating a shard file named :code:`shard-<shard_id>.arrow` for the current worker.
        - Converting dataset features to an Arrow schema.
        - Initializing the Arrow writer to stream data in Arrow format.

        The working directory is set to the save directory during the write process.

        Args:
            shard_id (int): The id of the shard being initialized.
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        worker_info = get_worker_info()
        # open shard file
        worker_info.ctx.file_path = f"shard-{shard_id:05}.arrow"
        worker_info.ctx.file = open(worker_info.ctx.file_path, "wb", buffering=0)
        # build arrow schema from dataset features
        assert info.features is not None
        worker_info.ctx.schema = info.features.arrow_schema
        # create arrow writer
        worker_info.ctx.writer = pa.ipc.new_stream(
            sink=worker_info.ctx.file, schema=worker_info.ctx.schema
        )

    def write_batch(self, batch: pa.Table) -> int:
        """Write a batch of samples to the Arrow shard.

        This method writes the samples as a record batch to the worker's shard file
        using the Arrow schema initialized during the :code:`initialize` phase.

        Args:
            batch (pa.Table): A list of samples to be written to the Arrow file.

        Returns:
            int: The number of bytes written to the shard.
        """
        info = get_worker_info()
        file_size = info.ctx.file.tell()
        # write sample to file
        info.ctx.writer.write(batch.cast(info.ctx.schema))
        # return bytes written to file
        return info.ctx.file.tell() - file_size

    def finalize_shard(self, info: DatasetInfo) -> None:
        """Finalize the writing process.

        This method closes the Arrow writer and the corresponding shard file.
        It ensures all written samples are properly flushed to disk.

        Args:
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        # close writer and file
        worker_info = get_worker_info()
        worker_info.ctx.writer.close()
        worker_info.ctx.file.close()
