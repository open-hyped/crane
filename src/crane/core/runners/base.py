"""Base Runners Module.

This module defines the base runner class and worker roles for data processing.
It includes an abstract base class for runners and an enumeration for worker roles
during multiprocessing.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable

from datasets import IterableDataset


class WorkerRole(str, Enum):
    """Enumeration of different roles a worker can assume during multiprocessing.

    Workers can dynamically switch between these roles based on the current processing stage
    and system needs.
    """

    STANDALONE = "standalone"
    """Role where the worker processes a data shard independently.

    In this role, the worker is responsible for both loading and processing a shard of the dataset
    without interacting with other workers.
    """

    PRODUCER = "producer"
    """Role where the worker produces data and adds it to a shared queue.

    In this role, the worker reads data from a shard and places it into the queue for further
    processing by other workers.
    """

    CONSUMER = "consumer"
    """Role where the worker consumes data from a shared queue for processing.

    In this role, the worker retrieves data from the queue (populated by a PRODUCER) and processes
    it.
    """


class WorkerProcessingStage(str, Enum):
    """Enumeration of different stages in the worker processing pipeline.

    Each stage represents a distinct phase in the data processing workflow.
    """

    STREAM = "stream"
    """Stage where the worker loads data from its data stream."""

    TRANSFORM = "transform"
    """Stage where the worker applies a transformations or workloads to the loaded data."""

    FINALIZE = "finalize"
    """Stage where the worker performs finalization tasks on the processed data."""


class BaseRunner(ABC):
    """Abstract base class for data processing runners.

    This class defines the interface for various types of runners that can process datasets.
    Subclasses must implement the :func:`run` method to provide specific data processing logic.
    """

    @abstractmethod
    def run(
        self,
        ds: IterableDataset,
        finalizer: Callable[[Any], Any],
        batch_size: None | int = None,
        formatting: None | str = None,
    ) -> None:
        """Execute data processing on the given dataset.

        Args:
            ds (IterableDataset): The dataset to process.
            finalizer (Callable[[Any], Any]): The function to apply to each sample or batch
                of samples in the dataset as the final processing step.
            batch_size (None | int): The size of each batch to process. If :code:`None`,
                process samples individually. Only affects the finalizer. Defaults to None.
            formatting (None | str): The data format in which samples or batches are provided
                to the finalizer function.
        """
        ...
