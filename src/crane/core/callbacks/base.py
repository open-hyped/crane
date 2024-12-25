"""Callback Module for Dataset Processing.

This module defines the :class:`Callback` base class and the :class:`CallbackManager`
class, which facilitate the implementation of callback mechanisms during dataset
processing. Callbacks allow users to define custom behavior for various stages of
processing, including starting, shard completion, and stopping of tasks.
"""

from datasets import IterableDataset

from ..monitor import ProgressMonitor


class Callback(object):
    """Base class for defining callback methods during dataset processing.

    This class provides a set of hooks that can be overridden to respond to various
    events during the processing of a dataset. Subclasses should implement these
    methods to define custom behavior at different stages of processing.
    """

    def on_start(self, monitor: ProgressMonitor, ds: IterableDataset) -> None:
        """Called when processing starts.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
            ds (IterableDataset): The dataset being processed.
        """
        ...  # pragma: not covered

    def on_shard_in_progress(self, monitor: ProgressMonitor, shard_id: int) -> None:
        """Called when a shard begins processing.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
            shard_id (int): The identifier of the shard currently being processed.
        """
        ...  # pragma: not covered

    def on_shard_completed(self, monitor: ProgressMonitor, shard_id: int) -> None:
        """Called when a shard has completed processing.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
            shard_id (int): The identifier of the completed shard.
        """
        ...  # pragma: not covered

    def on_shard_canceled(self, monitor: ProgressMonitor, shard_id: int) -> None:
        """Called when a shard is canceled.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
            shard_id (int): The identifier of the canceled shard.
        """
        ...  # pragma: not covered

    def on_stopping(self, monitor: ProgressMonitor) -> None:
        """Called when processing is being stopped.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
        """
        ...  # pragma: not covered

    def on_done(self, monitor: ProgressMonitor) -> None:
        """Called when processing is complete.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
        """
        ...  # pragma: not covered


class CallbackManager(Callback):
    """Manages multiple callbacks during dataset processing.

    This class acts as a wrapper around a list of callbacks, delegating calls to
    each registered callback method. It allows for the collective management of
    callbacks, ensuring that all registered callbacks respond to events in sync.
    """

    def __init__(self, callbacks: list[Callback]) -> None:
        """Initialize the callback manager.

        Args:
            callbacks (list[Callback]): A list of callback instances to manage.
        """
        self._callbacks = callbacks

    def on_start(self, monitor: ProgressMonitor, ds: IterableDataset) -> None:
        """Calls the on_start method of each registered callback.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
            ds (IterableDataset): The dataset being processed.
        """
        for callback in self._callbacks:
            callback.on_start(monitor, ds)

    def on_shard_in_progress(self, monitor: ProgressMonitor, shard_id: int) -> None:
        """Calls the on_shard_in_progress method of each registered callback.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
            shard_id (int): The identifier of the shard currently being processed.
        """
        for callback in self._callbacks:
            callback.on_shard_in_progress(monitor, shard_id)

    def on_shard_completed(self, monitor: ProgressMonitor, shard_id: int) -> None:
        """Calls the on_shard_completed method of each registered callback.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
            shard_id (int): The identifier of the completed shard.
        """
        for callback in self._callbacks:
            callback.on_shard_completed(monitor, shard_id)

    def on_shard_canceled(self, monitor: ProgressMonitor, shard_id: int) -> None:
        """Calls the on_shard_canceled method of each registered callback.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
            shard_id (int): The identifier of the canceled shard.
        """
        for callback in self._callbacks:
            callback.on_shard_canceled(monitor, shard_id)

    def on_stopping(self, monitor: ProgressMonitor) -> None:
        """Calls the on_stopping method of each registered callback.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
        """
        for callback in self._callbacks:
            callback.on_stopping(monitor)

    def on_done(self, monitor: ProgressMonitor) -> None:
        """Calls the on_done method of each registered callback.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress.
        """
        for callback in self._callbacks:
            callback.on_done(monitor)
