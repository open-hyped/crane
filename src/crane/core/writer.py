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
from typing import TypeAlias

import datasets
import pyarrow as pa

from .callbacks.base import Callback
from .consumer import DatasetConsumer
from .sharding import ShardingController, ShardingStrategy
from .utils import Compose, RunAll, chdir

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

    Subclasses should implement methods for writing individual samples, and initializing
    and finalizing the write operation. The working directory is temporarily set to the
    save directory for writing operations.
    """

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
            write_batch_size (int): The number of samples to write in a single batch.
                Defaults to 32.
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
        state["_data_files"] = [{"filename": fname} for fname in os.listdir(".")]

        # write state to directory
        with open(datasets.config.DATASET_STATE_JSON_FILENAME, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)

    def _write_dataset(
        self, ds: datasets.IterableDataset | datasets.Dataset, save_dir: str
    ) -> None:
        """Write the single dataset split to the specified directory.

        Args:
            ds (datasets.IterableDataset | datasets.Dataset): The dataset or iterable dataset to be
                written.
            save_dir (str): The directory where the split data will be saved.
        """
        logger.info(f"Writing dataset split {ds.split} to {os.getcwd()}.")

        if ds.info is None:
            warnings.warn(
                "The dataset has no metadata information (ds.info is None). "
                "Ensure that the dataset has been properly loaded and contains necessary "
                "information. Proceeding without this metadata may lead to incomplete or "
                "incorrect data writing.",
                UserWarning,
            )

        # create sharding controller
        sharding_controller = ShardingController(
            is_multi_processed=self._num_proc > 1,
            sharding_strategy=self._sharding_strategy,
            max_shard_size=self._max_shard_size,
            sample_size_key=self._sample_size_key,
            initialize_shard=partial(self.initialize_shard, info=ds.info),
            finalize_shard=partial(self.finalize_shard, info=ds.info),
        )

        # wrap write function in sharding callback if needed
        write_fn = (
            self.write_batch
            if not sharding_controller.is_active
            else Compose(
                sharding_controller.update,
                self.write_batch,
                sharding_controller.callback,
            )
        )

        os.makedirs(save_dir, exist_ok=True)
        # convert dataset to iterable dataset
        if isinstance(ds, datasets.Dataset):
            ds = ds.to_iterable_dataset(self._num_proc)

        with chdir(save_dir):
            # write dataset to directory
            consumer = DatasetConsumer(
                num_proc=self._num_proc,
                prefetch_factor=self._prefetch,
                on_start=RunAll(
                    sharding_controller.initialize,
                    partial(self.initialize, ds.info),
                ),
                on_finish=RunAll(
                    sharding_controller.finalize,
                    partial(self.finalize, ds.info),
                ),
                progress_report_interval=self._tqdm_update_interval,
                disable_tqdm=self._disable_tqdm,
                callbacks=self._callbacks,
            )
            consumer.consume(
                ds, finalizer=write_fn, batch_size=self._write_batch_size, formatting="arrow"
            )

            # write dataset info and state
            self._write_state(ds)
            self._write_info(ds)

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

    @abstractmethod
    def write_batch(self, batch: pa.Table) -> int:
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
        ...

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
