"""Sharding Management Module.

This module provides functionality for managing the sharding of datasets during the writing process.
It includes the :class:`ShardingController` class, which facilitates the implementation of various
sharding strategies based on sample counts, sample attributes, or file sizes. This allows for
efficient management of large datasets by dividing them into smaller, manageable shards.
"""

import logging
import multiprocessing as mp
import warnings
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable

import pyarrow as pa
import pyarrow.compute as pc

logger = logging.getLogger(__name__)


def _parse_size(size_str: str) -> int:
    """Convert a string representation of size to bytes.

    Args:
        size_str (str): The size string (e.g., '5GB', '200MB', '1.5KB').

    Returns:
        int: The size in bytes.

    Raises:
        ValueError: If the size_str is not a valid format.
    """
    size_str = size_str.strip().upper()
    size_units = OrderedDict(
        [("KB", 1024), ("MB", 1024**2), ("GB", 1024**3), ("TB", 1024**4), ("B", 1)]
    )

    if len(size_str) < 2 or not any(unit in size_str for unit in size_units):
        raise ValueError(f"Invalid size format: {size_str}")

    for unit, factor in size_units.items():
        if size_str.endswith(unit):
            try:
                size_value = float(size_str[: -len(unit)].strip())
                return int(size_value * factor)
            except ValueError as e:
                raise ValueError(f"Invalid size value: {size_str}") from e

    raise ValueError(f"Invalid size format: {size_str}")


class ShardingStrategy(str, Enum):
    """Enum representing different strategies for sharding dataset samples.

    The strategy determines how the dataset will be divided into shards based on
    either the number of samples, an attribute of the sample, or the file size
    of the written data.
    """

    SAMPLE_COUNT = "sample_count"
    """Shard based on the number of samples.

    In this mode, shards are created once a specified number of samples
    has been written. This is the default mode.
    """

    SAMPLE_ITEM = "sample_item"
    """Shard based on an attribute of each sample.

    In this mode, a specific item or attribute of the sample is used to determine
    shard size. The attribute to be measured must be specified as part of the sharding
    configuration (e.g., an attribute that represents the sample size in terms of data).
    """

    FILE_SIZE = "file_size"
    """Shard based on the total size of written files.

    In this mode, the size of the output files is monitored, and a new shard is started
    once a specified file size threshold is reached (e.g., 1 GB).
    """

    NONE = "none"
    """Disable sharding.

    In this mode, all samples are written to a single file or destination
    without splitting them into multiple shards.
    """


class ShardingController(object):
    """Controller responsible for managing dataset sharding during the writing process.

    This class provides an interface to manage sharding strategies, track shard sizes,
    and control the lifecycle of shards, such as initializing and finalizing shards
    when specific thresholds are met.
    """

    def __init__(
        self,
        is_multi_processed: bool,
        sharding_strategy: ShardingStrategy,
        max_shard_size: None | int | str,
        sample_size_key: None | str,
        initialize_shard: Callable[[int], Any],
        finalize_shard: Callable[[int], Any],
    ) -> None:
        """Initialize the :class:`ShardingController`.

        Args:
            is_multi_processed (bool): Whether the sharding controller will manage
                multiple processes.
            sharding_strategy (ShardingStrategy): The strategy for determining when to
                create a new shard.
            max_shard_size (None | int | str): The maximum size of a shard. The format
                depends on the sharding strategy (e.g., integer for sample counts or byte sizes,
                or a string for file size such as "5GB").
            sample_size_key (None | str): The key to measure sample size, only used with the
                :class:`SAMPLE_ITEM` strategy.
            initialize_shard (Callable[[int], Any]): A function to initialize a new shard.
            finalize_shard (Callable[[int], Any]): A function to finalize the current shard.
        """
        global _manager

        if (sharding_strategy is not ShardingStrategy.SAMPLE_ITEM) and (
            sample_size_key is not None
        ):
            warnings.warn(
                "The `sample_size_key` parameter is ignored when using sharding strategies "
                "other than SAMPLE_ITEM.",
                UserWarning,
            )

        if (sharding_strategy is ShardingStrategy.SAMPLE_ITEM) and (sample_size_key is None):
            raise ValueError(
                "The `sample_size_key` must be specified when using the `SAMPLE_ITEM` sharding "
                "strategy."
            )

        if (sharding_strategy is not ShardingStrategy.FILE_SIZE) and isinstance(
            max_shard_size, str
        ):
            raise ValueError(
                "The `max_shard_size` parameter must be an integer when not using the `FILE_SIZE` "
                "sharding strategy."
            )

        if isinstance(max_shard_size, str):
            # parse the size string to an integer
            max_shard_size = _parse_size(max_shard_size)

        self._is_multi_processed = is_multi_processed
        # sharding strategy
        self._sharding_strategy = sharding_strategy
        self._max_shard_size = max_shard_size
        self._sample_size_key = sample_size_key
        # shard state
        self._shard_id: None | int = None
        self._shard_size = 0
        self._shard_bytes = 0
        self._sample_size = 0
        # global shard state
        if is_multi_processed:
            manager = mp.Manager()
            self._num_shards = manager.Value("i32", 0)
            self._lock = manager.Lock()
        else:
            self._num_shards = 0
            self._lock = None
        # shard creation
        self._initialize_shard = initialize_shard
        self._finalize_shard = finalize_shard

    @property
    def is_active(self) -> bool:
        """Returns whether the sharding controller is active.

        Returns:
            bool: True if sharding is active, False otherwise.
        """
        return self._sharding_strategy is not ShardingStrategy.NONE

    def _next_shard_id(self) -> None:
        """Assign the next shard ID.

        If multi-processing is enabled, this is done with thread-safe increments
        using locks; otherwise, the counter is incremented directly.
        """
        assert self._shard_id is None

        if not self._is_multi_processed:
            self._shard_id = self._num_shards
            self._num_shards += 1

        else:
            with self._lock:
                self._shard_id = self._num_shards.get()
                self._num_shards.set(self._shard_id + 1)

    def _reset_state(self) -> None:
        """Reset the shard state variables for the next shard."""
        self._shard_id = None
        self._shard_size = 0
        self._shard_bytes = 0
        self._sample_size = 0

    def callback(self, batch: pa.Table) -> pa.Table:
        """Process each batch before writing and check if a new shard is required.

        Args:
            batch (pa.Table): The batch of samples.

        Returns:
            pa.Table: The batch, unchanged.
        """
        if self._sharding_strategy is ShardingStrategy.SAMPLE_ITEM:
            # cache the size of the sample to be used later in the shard size update
            self._sample_size = pc.sum(batch.column(self._sample_size_key)).as_py()

        # check if shard is full
        if self._shard_size >= self._max_shard_size:
            # finalize current shard and initialize a new one
            self.finalize()
            self.initialize()

        return batch

    def update(self, num_bytes: int) -> None:
        """Update the shard size based on the number of bytes written or sample size.

        Args:
            num_bytes (int): The number of bytes written to the current shard.
        """
        # udpate shard size according to the strategy
        self._shard_bytes += num_bytes
        self._shard_size += (
            1
            if self._sharding_strategy is ShardingStrategy.SAMPLE_COUNT
            else self._sample_size
            if self._sharding_strategy is ShardingStrategy.SAMPLE_ITEM
            else num_bytes
            if self._sharding_strategy is ShardingStrategy.FILE_SIZE
            else 0
        )

    def initialize(self) -> None:
        """Initialize a new shard by assigning an ID and invoking the initialization logic."""
        # initialize new shard
        self._next_shard_id()
        self._initialize_shard(self._shard_id)
        logger.info(f"Initialized new shard with id {self._shard_id}.")

    def finalize(self) -> None:
        """Finalize the current shard, reset its state, and invoke the finalization logic."""
        if self._shard_id is not None:
            self._finalize_shard()
            logger.info(f"Finalized shard with id {self._shard_id}")
            self._reset_state()
