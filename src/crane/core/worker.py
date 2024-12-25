"""Module for managing worker information in a multiprocessing context.

This module provides functionality to set and retrieve worker information,
such as rank, number of workers, and seed, which is useful for managing
worker-specific configurations in parallel processing setups.

This module optionally relies on PyTorch's worker information if the `torch`
package is installed and used in a multiprocessing context.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class WorkerInfo(object):
    """Holds information about the worker in a multiprocessing context."""

    rank: int
    """The rank or ID of the worker."""

    num_workers: int
    """The total number of workers."""

    seed: int
    """The seed used for random number generation in this worker."""

    ctx: SimpleNamespace = field(default_factory=SimpleNamespace)
    """A namespace for any additional worker context data."""


_worker_info: None | WorkerInfo = None


def get_worker_info() -> None | WorkerInfo:
    """Retrieves the current worker's information.

    This function will attempt to retrieve worker information from PyTorch
    if the 'torch' package is installed and the worker is part of a multiprocessing
    setup.

    Returns:
        WorkerInfo | None: The worker information, or None if the worker info is not set
        or if not running in a multiprocessing context.
    """
    global _worker_info

    return _worker_info


def set_worker_info(rank: int, num_workers: int, seed: int, **ctx: Any) -> WorkerInfo:
    """Sets the worker information for the current process.

    Args:
        rank (int): The rank of the worker.
        num_workers (int): The total number of workers.
        seed (int): The seed for random number generation in this worker.
        **ctx (Any): Additional context data to be stored in the worker's context (ctx).

    Returns:
        WorkerInfo: The newly created worker information.

    Raises:
        AssertionError: If worker information is already set, preventing reassignment.
    """
    global _worker_info

    # make sure the worker info is not set yet
    assert get_worker_info() is None, "Worker info already set."

    # set worker info
    _worker_info = WorkerInfo(rank, num_workers, seed, SimpleNamespace(**ctx))

    return _worker_info


def reset_worker_info() -> None:
    """Resets the worker information for the current process."""
    global _worker_info
    _worker_info = None
