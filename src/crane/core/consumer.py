"""Dataset Consumer Module.

This module defines the :class:`DatasetConsumer` class, which is responsible for consuming and
processing datasets. It provides a flexible framework for applying user-defined functions to
samples in a dataset while managing data pipelines and supporting both single-process and
multi-process execution.
"""

import logging
import multiprocessing as mp
from typing import Any, Callable

from datasets.iterable_dataset import IterableDataset, identity_func

from .callbacks.base import Callback, CallbackManager
from .callbacks.tqdm_reporter import TqdmReporterCallback
from .runners.base import BaseRunner
from .runners.main_process_runner import MainProcessRunner
from .runners.multi_process_runner import DynamicMultiprocessingRunner

logger = logging.getLogger(__name__)


def _do_nothing():
    """A no-operation function that returns nothing."""
    return  # pragma: not covered


class DatasetConsumer(object):
    """Consumes and processes a dataset.

    This class prepares a dataset for processing, manages a processing pipeline,
    and executes data consumption, optionally using parallel processing.
    """

    def __init__(
        self,
        num_proc: int = mp.cpu_count(),
        prefetch_factor: int = 128,
        on_start: Callable[[], Any] = _do_nothing,
        on_finish: Callable[[], Any] = _do_nothing,
        progress_report_interval: float = 0.5,
        disable_tqdm: bool = False,
        callbacks: list[Callback] = [],
    ) -> None:
        """Initialize the dataset consumer.

        Args:
            num_proc (int): The number of processes to use for parallel processing. Defaults to the
                number of CPUs. If set to 1, processing will be single-threaded.
            prefetch_factor (int, optional): The number of items to prefetch in the pipeline.
                Default is 8.
            on_start (Callable[[], Any], optional): Hook called on worker start.
            on_finish (Callable[[], Any], optional): Hook called on worker finish.
            progress_report_interval (float, optional): The interval in seconds at which the tqdm
                progress bar updates. Default is 0.1.
            disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Default is
                False, meaning the progress bar is enabled.
            callbacks (list[Callback]): A list of callback functions that will be invoked at
                various points during the data processing lifecycle.
        """
        if not disable_tqdm:
            callbacks = callbacks + [TqdmReporterCallback(progress_report_interval)]

        self._num_proc = num_proc
        self._prefetch = prefetch_factor

        self._on_start = on_start
        self._on_finish = on_finish

        self._report_interval = progress_report_interval
        self._callback = CallbackManager(callbacks)

    def consume(
        self,
        ds: IterableDataset,
        finalizer: Callable[[Any], Any] = identity_func,
        batch_size: None | int = None,
        formatting: None | str = None,
    ) -> None:
        """Process the dataset.

        Args:
            ds (IterableDataset): The dataset to process.
            finalizer (Callable[[Any], Any]): The function to apply to each sample or batch
                of samples in the dataset as the final processing step. Defaults to noop.
            batch_size (None | int): The size of each batch to process. If :code:`None`,
                process samples individually. Only affects the finalizer. Defaults to None.
            formatting (None | str): The data format in which samples or batches are provided
                to the finalizer function.
        """
        runner: BaseRunner

        if self._num_proc > 1:
            logger.info("Running in multi-process mode.")
            # create the multiprocessing runner and run it
            runner = DynamicMultiprocessingRunner(
                num_workers=self._num_proc,
                prefetch_factor=self._prefetch,
                worker_init=self._on_start,
                worker_finalize=self._on_finish,
                progress_report_interval=self._report_interval,
                callback=self._callback,
            )
        else:
            logger.info("Running in single-process mode.")
            # create the main process runner and run it
            runner = MainProcessRunner(
                batch_size=self._prefetch,
                env_init=self._on_start,
                env_finalize=self._on_finish,
                progress_report_interval=self._report_interval,
                callback=self._callback,
            )

        runner.run(ds, finalizer, batch_size, formatting)
