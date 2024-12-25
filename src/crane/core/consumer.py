"""Dataset Consumer Module.

This module defines the :class:`DatasetConsumer` class, which is responsible for consuming and
processing datasets. It provides a flexible framework for applying user-defined functions to
samples in a dataset while managing data pipelines and supporting both single-process and
multi-process execution.
"""

import logging
import multiprocessing as mp
from typing import Any, Callable

from datasets import IterableDataset

from .callbacks.base import Callback, CallbackManager
from .callbacks.tqdm_reporter import TqdmReporterCallback
from .runners.main_process_runner import MainProcessRunner
from .runners.multi_process_runner import DynamicMultiprocessingRunner
from .utils import Sample

logger = logging.getLogger(__name__)


def _do_nothing():
    """A no-operation function that returns nothing."""
    return


class DatasetConsumer(object):
    """Consumes and processes a dataset.

    This class prepares a dataset for processing, manages a processing pipeline,
    and executes data consumption, optionally using parallel processing.
    """

    def __init__(
        self,
        fn: Callable[[Sample], Any],
        num_proc: int = mp.cpu_count(),
        prefetch_factor: int = 128,
        initialize: Callable[[], Any] = _do_nothing,
        finalize: Callable[[], Any] = _do_nothing,
        progress_report_interval: float = 0.5,
        disable_tqdm: bool = False,
        callbacks: list[Callback] = [],
    ) -> None:
        """Initialize the dataset consumer.

        Args:
            fn (Callable[[Sample], Any]): The function to be applied to each sample in the dataset.
            num_proc (int): The number of processes to use for parallel processing. Defaults to the
                number of CPUs. If set to 1, processing will be single-threaded.
            prefetch_factor (int, optional): The number of items to prefetch in the pipeline.
                Default is 8.
            initialize (Callable[[], Any], optional): A callable to initialize the processing
                pipeline. Default is a no-op function.
            finalize (Callable[[], Any], optional): A callable to finalize the processing
                pipeline. Default is a no-op function.
            progress_report_interval (float, optional): The interval in seconds at which the tqdm
                progress bar updates. Default is 0.1.
            disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Default is
                False, meaning the progress bar is enabled.
            callbacks (list[Callback]): A list of callback functions that will be invoked at
                various points during the data processing lifecycle.
        """
        if not disable_tqdm:
            callbacks = callbacks + [TqdmReporterCallback(progress_report_interval)]

        self._fn = fn
        self._num_proc = num_proc
        self._prefetch = prefetch_factor

        self._initialize = initialize
        self._finalize = finalize

        self._report_interval = progress_report_interval
        self._callback = CallbackManager(callbacks)

    def consume(self, ds: IterableDataset) -> None:
        """Process the dataset.

        Args:
            ds (IterableDataset): The dataset to process.
        """
        if self._num_proc > 1:
            logger.info("Running in multi-process mode.")
            # create the multiprocessing runner and run it
            runner = DynamicMultiprocessingRunner(
                num_workers=self._num_proc,
                prefetch_factor=self._prefetch,
                worker_init=self._initialize,
                worker_finalize=self._finalize,
                progress_report_interval=self._report_interval,
                callback=self._callback,
            )
            runner.run(ds, self._fn)

        else:
            logger.info("Running in single-process mode.")
            # create the main process runner and run it
            runner = MainProcessRunner(
                batch_size=self._prefetch,
                initialize=self._initialize,
                finalize=self._finalize,
                progress_report_interval=self._report_interval,
                callback=self._callback,
            )
            runner.run(ds, self._fn)
