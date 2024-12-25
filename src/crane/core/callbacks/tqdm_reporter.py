"""Tqdm progress monitoring callback for dataset processing using tqdm.

This module includes the :class:`TqdmReporterCallback` class, which integrates with a
:class:`ProgressMonitor` to visualize the real-time progress of dataset processing tasks.
"""
import logging
import threading
from collections import Counter
from typing import Any

from datasets import IterableDataset
from tqdm.auto import tqdm

from ..monitor import ProgressMonitor
from ..runners.base import WorkerRole
from .base import Callback

logger = logging.getLogger(__name__)


def _run_tqdm(monitor: ProgressMonitor, update_interval: float, **kwargs: Any) -> None:
    """Runs a tqdm progress bar to display the current status and progress.

    This function is executed in a separate thread to monitor and visualize the progress of
    a dataset processing task. It updates the progress bar based on the number of completed
    shards and processed samples, while displaying the current throughput and the status
    of active workers.

    Args:
        monitor (ProgressMonitor): The object tracking the status of workers and the processing
            task.
        update_interval (float): The interval in seconds between progress bar updates.
        kwargs (Any): Additional keyword arguments passed to the :func:`tqdm` function.
    """
    logger.info("Thread started.")

    def build_desc_str():
        counts = Counter(monitor._roles)
        return (
            f"Workers {len(monitor.alive_workers)}/{monitor.num_workers} "
            f"(S={counts[WorkerRole.STANDALONE]}, "
            f"P={counts[WorkerRole.PRODUCER]}, "
            f"C={counts[WorkerRole.CONSUMER]})"
        )

    with tqdm(total=monitor._num_shards, desc=build_desc_str(), unit="sh", **kwargs) as pbar:
        logger.debug(f"Initialized tqdm progress bar with {monitor.num_shards} total shards.")

        def _iter():
            # iterate as long as the tracker is running
            while not monitor._done.wait(timeout=update_interval):
                yield
            # do a final update after the tracker finished
            yield

        for _ in _iter():
            # get current total samples and throughput
            total_samples = monitor.num_processed_samples
            throughput = monitor.samples_per_second
            # log
            logger.debug(
                f"Updating progress: Total samples {total_samples}, "
                f"Throughput {throughput:.02f}it/s"
            )
            # format total samples
            formatting_string = "%d" if total_samples < 10**6 else "%.2e"
            formatted_total_samples = formatting_string % total_samples
            # update progress bar
            pbar.set_postfix_str(
                (
                    f"{monitor.num_buffered_samples}q, "
                    f"{throughput:.02f}it/s, "
                    f"{formatted_total_samples}it"
                ),
                refresh=False,
            )

            # update the progress bar
            pbar.set_description(build_desc_str(), refresh=True)
            pbar.update(len(monitor.completed_shards) - pbar.n)

    logger.info("Thread finished.")


class TqdmReporterCallback(Callback):
    """A callback class that manages a tqdm progress bar to monitor the progress of a task.

    This callback is designed to be used with a :class:`ProgressMonitor` to visualize progress
    in real-time during dataset processing. It spawns a background thread that continuously
    updates the progress bar until the task is complete.
    """

    def __init__(self, update_interval: float = 0.1, **kwargs: Any) -> None:
        """Initializes the TqdmReporterCallback with the update interval.

        Args:
            update_interval (float): The interval in seconds between progress bar updates.
            **kwargs (Any): Additional keyword arguments that are forwarded to the tqdm
                progress bar constructor. These can include options such as :code:`ncols`,
                :code:`miniters`, :code:`ascii`, etc. For more information please refer to
                the official tqdm documentation.
        """
        self._update_interval = update_interval
        self._tqdm_kwargs = kwargs
        self._thread: None | threading.Thread = None

    def on_start(self, monitor: ProgressMonitor, ds: IterableDataset) -> None:
        """Starts the progress monitoring thread when the task begins.

        This method is called when the dataset processing task starts. It spawns a
        new thread that runs the tqdm progress bar, using any additional arguments
        passed to customize tqdm's behavior.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress and worker states.
            ds (IterableDataset): The dataset being processed.
        """
        assert self._thread is None
        self._thread = threading.Thread(
            target=_run_tqdm,
            args=(monitor, self._update_interval),
            kwargs=self._tqdm_kwargs,
            daemon=True,
        )
        self._thread.start()

    def on_done(self, monitor: ProgressMonitor) -> None:
        """Joins the progress monitoring thread when the task is complete.

        This method is called when the dataset processing task finishes. It ensures that
        the progress monitoring thread is properly joined before exiting.

        Args:
            monitor (ProgressMonitor): The monitor tracking the task's progress and worker states.
        """
        if self._thread is not None:
            self._thread.join()
