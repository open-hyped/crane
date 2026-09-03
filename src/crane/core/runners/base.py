"""Base Runners Module.

This module defines the base runner class and worker roles for data processing.
It includes an abstract base class for runners, an enumeration for worker roles
during multiprocessing, and how a runner reacts to a workload that raises.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, NamedTuple

from datasets import IterableDataset


class FailurePolicy(str, Enum):
    """What a runner does when the workload raises on a shard."""

    FAIL_FAST = "fail_fast"
    """Stop the run and raise :class:`ShardProcessingError`.

    The default. A run whose workload raised has not produced the dataset that was asked
    for, and failing loudly is better than returning a partial one that looks complete.
    """

    SKIP_SHARD = "skip_shard"
    """Log the failure, drop that shard, and carry on with the rest.

    For a corpus where a few bad shards are expected and worth losing. The ids that failed
    are collected and raised at the end as :class:`ShardProcessingError` unless the caller
    inspects them - see :attr:`ShardProcessingError.failures`.
    """


class ShardFailure(NamedTuple):
    """One workload failure, as reported by the worker that hit it."""

    rank: int
    """The worker that raised."""

    shard_id: None | int
    """The shard it was processing, if it held one."""

    error_type: str
    """The exception's class name."""

    error_message: str
    """Its message."""

    stack_trace: str
    """The formatted traceback, from inside the worker."""

    def __str__(self) -> str:
        """One-line summary, without the traceback."""
        return (
            f"worker {self.rank} on shard {self.shard_id}: "
            f"{self.error_type}: {self.error_message}"
        )


class ShardProcessingError(Exception):
    """Raised when the workload failed on one or more shards.

    Attributes:
        failures (list[ShardFailure]): Every failure that was reported, in the order they
            arrived. The first one's traceback is the most useful; the rest are often
            the same fault hitting every worker.
    """

    def __init__(self, failures: list["ShardFailure"]) -> None:
        """Initialize the error from the failures collected during the run.

        Args:
            failures (list[ShardFailure]): The reported failures.
        """
        self.failures = failures
        shards = sorted({f.shard_id for f in failures if f.shard_id is not None})
        summary = (
            f"Processing failed on {len(failures)} occasion(s), shards {shards}."
            if shards
            else f"Processing failed on {len(failures)} occasion(s)."
        )
        super().__init__(f"{summary}\nFirst failure:\n{failures[0].stack_trace}")


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
