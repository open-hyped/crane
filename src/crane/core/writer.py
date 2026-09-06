"""Base class for writing datasets to disk.

This module defines the :class:`BaseDatasetWriter` class, which serves as an abstract base
for writing datasets to disk in a structured and efficient manner. It facilitates
the saving of datasets in a format compatible with Hugging Face's `datasets` library.
"""

import json
import logging
import multiprocessing as mp
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict
from functools import partial
from typing import Any, Callable, ClassVar, TypeAlias

import datasets
import pyarrow as pa
from datasets.iterable_dataset import (
    ArrowExamplesIterable,
    BufferShuffledExamplesIterable,
    RebatchedArrowExamplesIterable,
    SelectColumnsIterable,
    SkipExamplesIterable,
    StepExamplesIterable,
    TakeExamplesIterable,
)

from .batching import BatchBuffer
from .callbacks.base import Callback
from .consumer import DatasetConsumer
from .runners.base import FailurePolicy
from .sharding import ShardingController, ShardingStrategy
from .utils import Compose, FormatType, RunAll, chdir
from .worker import get_worker_info

logger = logging.getLogger(__name__)


DatasetType: TypeAlias = (
    datasets.Dataset
    | datasets.DatasetDict
    | datasets.IterableDataset
    | datasets.IterableDatasetDict
)
"""Dataset Type Alias.

Alias for the different dataset types supported, including:

- :class:`Dataset`: A single dataset containing features and samples.
- :class:`DatasetDict`: A dictionary-like structure containing multiple datasets, often split into
  training, validation, and test sets.
- :class:`IterableDataset`: A dataset that is lazily loaded, allowing for streaming data processing.
- :class:`IterableDatasetDict`: A dictionary-like structure containing multiple iterable datasets.

This type alias is used to represent any of the aforementioned dataset types when processing or
consuming datasets in various contexts.
"""


class BaseDatasetWriter(ABC):
    """Base class for writing datasets to disk.

    This class provides a framework for saving a dataset to a specified directory.
    The folder structure and save format follows the Hugging Face :func:`save_to_disk`
    structure, ensuring compatibility with datasets saved using this format.

    Subclasses must implement at least one of the format-specific write methods
    (:func:`write_batch_py` or :func:`write_batch_arrow`). When writing a batch,
    the writer automatically routes to the best fitting supported function based
    on the format of the dataset being written to, minimizing format conversion
    overhead.
    """

    SUPPORTED_FORMATS: ClassVar[dict[FormatType, Callable[[Any], int]]] = dict()

    def __init_subclass__(cls, **kwargs):
        """Check for supported formats."""
        super().__init_subclass__(**kwargs)
        cls.SUPPORTED_FORMATS = {}
        # Detect supported formats by checking if the methods are overridden
        if cls.write_batch_py != BaseDatasetWriter.write_batch_py:
            cls.SUPPORTED_FORMATS[FormatType.PYTHON] = cls.write_batch_py
        if cls.write_batch_arrow != BaseDatasetWriter.write_batch_arrow:
            cls.SUPPORTED_FORMATS[FormatType.ARROW] = cls.write_batch_arrow
        # Ensure at least one method is implemented
        if len(cls.SUPPORTED_FORMATS) == 0:
            raise TypeError(
                f"`{cls.__name__}` must override at least one of "
                "`write_batch_py` or `write_batch_arrow`."
            )

    def __init__(
        self,
        save_dir: str,
        overwrite: bool = False,
        num_proc: int = mp.cpu_count(),
        prefetch_factor: int = 128,
        write_batch_size: int = 256,
        tqdm_update_interval: float = 0.5,
        disable_tqdm: bool = False,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FILE_SIZE,
        max_shard_size: None | int | str = "5GB",
        sample_size_key: None | str = None,
        callbacks: list[Callback] = [],
        failure_policy: FailurePolicy = FailurePolicy.FAIL_FAST,
    ) -> None:
        """Initialize the :class:`BaseDatasetWriter`.

        Args:
            save_dir (str): Directory where the dataset will be saved.
            overwrite (bool): Whether to overwrite existing files in the save directory.
                Defaults to False.
            num_proc (int): Number of processes to use for dataset processing. Defaults
                to the number of CPU cores.
            prefetch_factor (int): Number of samples to prefetch for improved
                performance. Defaults to 8.
            write_batch_size (int): The number of samples handed to the writer in one call,
                and so the unit the output is laid out in. Batches are regrouped to exactly
                this size before they reach the writer, whatever size the pipeline delivered:
                a pipeline's batches are sized for throughput and rebatched by every
                :func:`map` along the way, so without that the layout of the output would be
                an accident of the pipeline. Writing a batch per sample costs the writer
                nothing and costs every reader afterwards - a dataset written that way took
                125 seconds to open rather than 0.09. Defaults to 256.

                Interacts with :code:`max_shard_size`: a shard rolls over on what has reached
                the file, and nothing reaches it until a batch is full, so a shard can
                overshoot by up to one batch and a dataset smaller than one batch is written
                as a single shard. The rows held are held in memory, so a very wide row wants
                a smaller value.
            tqdm_update_interval (float): The interval in seconds at which the tqdm
                progress bar updates. Default is 0.1.
            disable_tqdm (bool): Whether to disable the tqdm progress bar. Default is
                False, meaning the progress bar is enabled.
            sharding_strategy (ShardingStrategy): The strategy to use for sharding
                dataset samples. Defaults to :class:`FILE_SIZE`.
            max_shard_size (None | int | str): Maximum size for each shard according to the
                sharding strategy. If specified, the sharding strategy will consider this limit.
                Defaults to '5GB' matching the default sharding strategy.
            sample_size_key (None | str): The key in the dataset sample to measure size if using
                the :class:`SAMPLE_ITEM` sharding strategy.
            callbacks (list[Callback]): List of callbacks.
            failure_policy (FailurePolicy): What to do when the workload raises on a shard.
                Defaults to :attr:`FailurePolicy.FAIL_FAST`, which stops the run and raises
                :class:`ShardProcessingError` rather than leaving a partial dataset that
                looks complete.
        """
        self.save_dir = save_dir
        self._overwrite = overwrite
        self._num_proc = num_proc
        self._prefetch = prefetch_factor
        self._write_batch_size = write_batch_size
        # tqdm setup
        self._tqdm_update_interval = tqdm_update_interval
        self._disable_tqdm = disable_tqdm
        # sharding
        self._sharding_strategy = sharding_strategy
        self._max_shard_size = max_shard_size
        self._sample_size_key = sample_size_key
        # callbacks
        self._callbacks = callbacks
        self._failure_policy = failure_policy

    def _open_shard(self, shard_id: int, info: datasets.DatasetInfo) -> None:
        """Start a shard, and the buffer that regroups what is written to it.

        The buffer belongs to the worker rather than to the writer, alongside the shard's
        own file handle. A worker receives its own copy of the writer for writing and
        another for finalizing - they travel to it separately - so a buffer held on the
        writer would be filled by one copy and flushed by the other, which is to say never.

        Args:
            shard_id (int): The id of the shard being opened.
            info (datasets.DatasetInfo): Information about the dataset being written.
        """
        worker = get_worker_info()
        worker.ctx.buffer = BatchBuffer(self._write_batch_size)
        worker.ctx.write_fn = None
        self.initialize_shard(shard_id, info)

    def _close_shard(self, info: datasets.DatasetInfo) -> None:
        """Write whatever the buffer still holds, then let the writer close the shard.

        The flush has to happen here rather than at the end of the run: a shard rolls over
        while there is still data to come, and rows held past that point would be written
        into the shard that follows - or, for the last shard, into nothing at all.

        Args:
            info (datasets.DatasetInfo): Information about the dataset being written.
        """
        worker = get_worker_info()
        remainder = worker.ctx.buffer.take()
        if (remainder is not None) and (worker.ctx.write_fn is not None):
            worker.ctx.write_fn(remainder)

        self.finalize_shard(info)

    def _write_batches(self, write_fn: Callable[[Any], int], batch: Any) -> int:
        """Hand the writer batches of the size it asked for, whatever size arrived.

        A pipeline's batches are sized for throughput and rebatched by every :func:`map`
        along the way, so what arrives here is not what :code:`write_batch_size` asked for.
        Passing it straight through makes that accident the layout of the file - a batch per
        row, if that is what the pipeline happened to produce - which costs nothing to write
        and is expensive for every reader of the file afterwards.

        Args:
            write_fn (Callable[[Any], int]): The writer's own batch write, bound by
                :func:`_get_write_fn`.
            batch (Any): The batch that arrived.

        Returns:
            int: The number of bytes written, which is zero while the buffer is still short
            of a full batch.
        """
        # Kept for `_close_shard`, which flushes without having a batch of its own to pass.
        worker = get_worker_info()
        worker.ctx.write_fn = write_fn
        return sum(write_fn(full) for full in worker.ctx.buffer.add(batch))

    def _write_info(self, ds: datasets.IterableDataset) -> None:
        """Write dataset information to a JSON file in the save directory.

        Args:
            ds (datasets.IterableDataset): The dataset object containing metadata to be saved.
        """
        logger.info(f"Writing dataset info to {os.getcwd()}.")
        info = asdict(ds.info)

        with open(
            datasets.config.DATASET_INFO_FILENAME, "w", encoding="utf-8"
        ) as dataset_info_file:
            # Sort only the first level of keys, or we might shuffle fields of nested
            # features if we use sort_keys=True
            sorted_keys_dataset_info = {key: info[key] for key in sorted(info)}
            json.dump(sorted_keys_dataset_info, dataset_info_file, indent=2)

    def _write_state(self, ds: datasets.IterableDataset) -> None:
        """Write the state of the dataset to a JSON file in the save directory.

        Args:
            ds (datasets.IterableDataset): The dataset object containing state information to be
                saved.
        """
        logger.info(f"Writing dataset state to {os.getcwd()}.")

        keys = (
            "_fingerprint",
            "_format_columns",
            "_format_kwargs",
            "_format_type",
            "_output_all_columns",
        )
        # build state
        state = {key: getattr(ds, key, None) for key in keys}
        state["_format_kwargs"] = {}
        state["_split"] = str(ds.split) if ds.split is not None else ds.split
        # sorted, so the shards are listed in their numbering rather than in whatever order
        # the filesystem happens to report, and files only, so a working directory left in
        # the save directory is not mistaken for a shard
        state["_data_files"] = [
            {"filename": fname} for fname in sorted(os.listdir(".")) if os.path.isfile(fname)
        ]

        # write state to directory
        with open(datasets.config.DATASET_STATE_JSON_FILENAME, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)

    def _get_write_fn(self, ds: datasets.IterableDataset) -> tuple[str, Callable]:
        """Determine the best format and corresponding write function for the dataset.

        Args:
            ds (datasets.IterableDataset): The dataset or iterable dataset to be written.

        Returns:
            tuple[str, Callable]: The determined formatting and the bound write function
                corresponding to the formatting.
        """
        ex_iterable = ds._ex_iterable
        # skip all operators that have no affect on the underlying data format
        while isinstance(
            ex_iterable,
            (
                SelectColumnsIterable,
                StepExamplesIterable,
                BufferShuffledExamplesIterable,
                SkipExamplesIterable,
                TakeExamplesIterable,
            ),
        ):
            ex_iterable = ex_iterable.ex_iterable
        # use arrow format if the underlying iterable yields arrow tables
        if isinstance(ds._ex_iterable, (ArrowExamplesIterable, RebatchedArrowExamplesIterable)):
            ds = ds.with_format(type=FormatType.ARROW.value)

        all_formats = [item.value for item in FormatType]
        supported_formats = type(self).SUPPORTED_FORMATS

        # Get the fallback formatting in case the dataset formatting is not supported
        fallback_formatting = next(iter(supported_formats.keys()))
        # Get the dataset formatting
        formatting = ds._formatting
        formatting = formatting.format_type if formatting is not None else fallback_formatting
        formatting = FormatType(formatting) if formatting in all_formats else fallback_formatting
        # Check if the formatting is supported by the writer
        formatting = formatting if formatting in supported_formats else fallback_formatting
        write_fn = supported_formats[formatting]
        return formatting, write_fn.__get__(self, type(self))

    def _write_dataset(
        self, ds: datasets.IterableDataset | datasets.Dataset, save_dir: str
    ) -> None:
        """Write the single dataset split to the specified directory.

        Args:
            ds (datasets.IterableDataset | datasets.Dataset): The dataset or iterable dataset to be
                written.
            save_dir (str): The directory where the split data will be saved.
        """
        if ds.info is None:
            warnings.warn(
                "The dataset has no metadata information (ds.info is None). "
                "Ensure that the dataset has been properly loaded and contains necessary "
                "information. Proceeding without this metadata may lead to incomplete or "
                "incorrect data writing.",
                UserWarning,
            )

        # convert dataset to iterable dataset
        if isinstance(ds, datasets.Dataset):
            ds = ds.to_iterable_dataset(self._num_proc)

        formatting, write_fn = self._get_write_fn(ds)
        logger.info(
            f"Routing write operation to {formatting} implementation ({write_fn.__qualname__})."
        )

        # create sharding controller
        sharding_controller = ShardingController(
            is_multi_processed=self._num_proc > 1,
            sharding_strategy=self._sharding_strategy,
            max_shard_size=self._max_shard_size,
            sample_size_key=self._sample_size_key,
            initialize_shard=partial(self._open_shard, info=ds.info),
            finalize_shard=partial(self._close_shard, info=ds.info),
            formatting=formatting,
        )

        # The controller opens a shard on the first batch a worker actually writes, so it has
        # to see every batch - including under the NONE strategy, where it opens the single
        # shard but never rolls over.
        write_fn = Compose(
            sharding_controller.update,
            partial(self._write_batches, write_fn),
            sharding_controller.callback,
        )

        logger.info(f"Writing dataset split {ds.split} to {os.getcwd()}.")

        os.makedirs(save_dir, exist_ok=True)
        with chdir(save_dir):
            # write dataset to directory
            consumer = DatasetConsumer(
                num_proc=self._num_proc,
                prefetch_factor=self._prefetch,
                # no shard is opened here: a worker that never receives data must not leave
                # an empty shard file behind, so the controller opens one on first write
                on_start=partial(self.initialize, ds.info),
                on_finish=RunAll(
                    sharding_controller.finalize,
                    partial(self.finalize, ds.info),
                ),
                progress_report_interval=self._tqdm_update_interval,
                disable_tqdm=self._disable_tqdm,
                callbacks=self._callbacks,
                failure_policy=self._failure_policy,
            )
            consumer.consume(
                ds, finalizer=write_fn, batch_size=self._write_batch_size, formatting=formatting
            )

            # write dataset info and state
            self._write_state(ds)
            self._write_info(ds)
            # give the writer a look at the finished dataset as a whole
            self.finalize_dataset(ds)

    def finalize_dataset(self, ds: datasets.IterableDataset) -> None:
        """Called once after every shard of a split has been written.

        Runs in the main process, after :func:`_write_state` and :func:`_write_info`, with
        the working directory set to the split's save directory. Unlike
        :func:`finalize_shard`, which each worker runs for its own shard, this is the only
        point at which the complete set of shards exists and is visible to one process -
        the place for anything needing a whole-dataset view, such as an index, a manifest
        or a checksum.

        Does nothing by default.

        Args:
            ds (datasets.IterableDataset): The dataset that was written.
        """
        return  # pragma: not covered

    def write(self, ds: DatasetType) -> None:
        """Write the entire dataset or dataset dictionary to disk.

        Args:
            ds (DatasetType): The dataset or dataset dictionary to be written.
        """
        # check if save directory already exists
        if os.path.exists(self.save_dir):
            if not self._overwrite:
                raise FileExistsError(
                    f"Output path `{self.save_dir}` already exists. "
                    f"Set `overwrite=True` to overwrite."
                )
            else:
                # delete existing directory
                logger.info("Deleting existing directory: {self.save_dir}.")
                shutil.rmtree(self.save_dir)

        # create the save directory
        os.makedirs(self.save_dir, exist_ok=False)

        if isinstance(ds, (datasets.DatasetDict, datasets.IterableDatasetDict)):
            # write all splits
            for key, split in ds.items():
                self._write_dataset(split, os.path.join(self.save_dir, key))
            # write dataset splits json
            with open(
                os.path.join(self.save_dir, datasets.config.DATASETDICT_JSON_FILENAME), "w+"
            ) as f:
                f.write(json.dumps({"splits": list(ds.keys())}))

        else:
            # save dataset to directory
            self._write_dataset(ds, self.save_dir)

    def write_batch_py(self, batch: dict[str, list[Any]]) -> int:
        """Abstract method for writing a batch of samples in Python-native format.

        This method writes a batch of samples to the dataset shard and returns the
        number of bytes written.

        The working directory is temporarily set to the save directory during this method,
        and any files created will be saved in the designated dataset directory.

        Args:
            batch (dict[str, list[Any]]): The batch of samples, where keys are column names
                and values are lists of column data.

        Returns:
            int: The number of bytes written to the shard.
        """
        raise NotImplementedError()

    def write_batch_arrow(self, batch: pa.Table) -> int:
        """Abstract method for writing a batch of samples.

        This method writes a batch of samples to the dataset shard and returns the
        number of bytes written.

        The working directory is temporarily set to the save directory during this method,
        any files created will be saved in the designated dataset directory.

        Args:
            batch (pa.Table): The batch of samples in pyarrow table format.

        Returns:
            int: The number of bytes written to the shard.
        """
        raise NotImplementedError()

    def initialize(self, info: datasets.DatasetInfo) -> None:
        """Initialize the global dataset write process.

        This method is responsible for any setup tasks that need to be performed once before
        writing begins for the dataset. This could include setting up metadata files, preparing
        the global output directory, or initializing any resources required for the write
        operation. The working directory is temporarily set to the global directory during
        this method.

        Args:
            info (datasets.DatasetInfo): Information about the dataset to be written, including
                metadata and configuration details.
        """
        return  # pragma: not covered

    def finalize(self, info: datasets.DatasetInfo) -> None:
        """Finalize the global dataset write process.

        This method is responsible for any cleanup tasks or final operations that should be
        performed after all shards of the dataset have been processed and written to disk.
        This could include writing final metadata files, closing any global resources, and
        ensuring that all data is properly stored. The working directory is temporarily set
        to the global save directory during this method.

        Args:
            info (datasetsDatasetInfo): Information about the dataset that was written, including
                metadata and configuration details.
        """
        return  # pragma: not covered

    @abstractmethod
    def initialize_shard(self, shard_id: int, info: datasets.DatasetInfo) -> None:
        """Abstract method for initializing the write process for a new shard.

        Any setup tasks specific to writing a new shard, such as creating necessary files or folders
        for the shard, should take place here. The working directory is temporarily set to the save
        directory during this method.

        Args:
            shard_id (int): The id of the shard being initialized.
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        ...

    @abstractmethod
    def finalize_shard(self, info: datasets.DatasetInfo) -> None:
        """Abstract method for finalizing the write process for the current shard.

        This method should handle any cleanup or final write operations after the samples for the
        current shard have been processed. The working directory is temporarily set to the save
        directory during this method.

        Args:
            info (DatasetInfo): Information about the dataset to be written, including metadata
                and configuration details.
        """
        ...
