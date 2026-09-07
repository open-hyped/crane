"""Dataset Consumer Module.

This module defines the :class:`DatasetConsumer` class, which is responsible for consuming and
processing datasets. It provides a flexible framework for applying user-defined functions to
samples in a dataset while managing data pipelines and supporting both single-process and
multi-process execution.
"""

import logging
import multiprocessing as mp
from typing import TYPE_CHECKING, Any, Callable

from datasets.iterable_dataset import IterableDataset, identity_func

from .callbacks.base import Callback, CallbackManager
from .callbacks.tqdm_reporter import TqdmReporterCallback
from .runners.base import BaseRunner, FailurePolicy
from .runners.main_process_runner import MainProcessRunner
from .runners.multi_process_runner import DynamicMultiprocessingRunner

if TYPE_CHECKING:  # pragma: not covered
    from ..distributed.core.base import DistributedBackend
    from ..distributed.core.run import DistributedRun

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
        num_proc: None | int = None,
        prefetch_factor: int = 128,
        on_start: Callable[[], Any] = _do_nothing,
        on_finish: Callable[[], Any] = _do_nothing,
        progress_report_interval: float = 0.5,
        disable_tqdm: bool = False,
        callbacks: list[Callback] = [],
        failure_policy: FailurePolicy = FailurePolicy.FAIL_FAST,
    ) -> None:
        """Initialize the dataset consumer.

        Args:
            num_proc (None | int): The number of processes to use for parallel processing.
                Defaults to :code:`None`, which means the number of CPUs locally, and the
                cores the allocation reserved for a run started with :func:`submit`. If set
                to 1, processing will be single-threaded.
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
            failure_policy (FailurePolicy): What to do when the workload raises on a shard.
                Defaults to stopping the run and raising :class:`ShardProcessingError`.
        """
        if not disable_tqdm:
            callbacks = callbacks + [TqdmReporterCallback(progress_report_interval)]

        self._num_proc = num_proc
        self._prefetch = prefetch_factor
        self._failure_policy = failure_policy

        self._on_start = on_start
        self._on_finish = on_finish

        self._report_interval = progress_report_interval
        self._callback = CallbackManager(callbacks)

    def add_callback(self, callback: Callback) -> None:
        """Register another callback, on top of those given at construction.

        Args:
            callback (Callback): The callback to add.
        """
        self._callback.add(callback)

    @property
    def num_proc(self) -> int:
        """The number of processes this consumer uses, resolved.

        :code:`num_proc` is left unset by a caller who wants it to follow whatever the run
        is given - the machine's cores locally, or the cores a distributed job reserved.
        """
        return self._num_proc if self._num_proc is not None else mp.cpu_count()

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

        if self.num_proc > 1:
            logger.info("Running in multi-process mode.")
            # create the multiprocessing runner and run it
            runner = DynamicMultiprocessingRunner(
                num_workers=self.num_proc,
                prefetch_factor=self._prefetch,
                worker_init=self._on_start,
                worker_finalize=self._on_finish,
                progress_report_interval=self._report_interval,
                callback=self._callback,
                failure_policy=self._failure_policy,
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
                failure_policy=self._failure_policy,
            )

        runner.run(ds, finalizer, batch_size, formatting)

    def submit(
        self,
        ds: IterableDataset,
        *,
        on: "DistributedBackend",
        finalizer: Callable[[Any], Any] = identity_func,
        batch_size: None | int = None,
        formatting: None | str = None,
    ) -> "DistributedRun":
        """Process the dataset on a cluster, without waiting for it to finish.

        The dataset's shards are split across :code:`on.num_jobs` jobs, each processing its
        share with the same runner a local run uses.

        Unlike :func:`consume`, this returns as soon as the work is queued:

        .. code-block:: python

            run = consumer.submit(ds, on=Slurm(num_jobs=16, work_dir="..."), finalizer=fn)
            run.wait()

        A consumer writes no dataset of its own, so there is no output directory for the
        run to keep its working files beside; the backend needs a :code:`work_dir`.

        Args:
            ds (IterableDataset): The dataset to process.
            on (DistributedBackend): Where to run.
            finalizer (Callable[[Any], Any]): The function to apply to each sample or batch
                of samples in the dataset as the final processing step.
            batch_size (None | int): The size of each batch to process. If :code:`None`,
                process samples individually. Only affects the finalizer.
            formatting (None | str): The data format in which samples or batches are
                provided to the finalizer function.

        Returns:
            DistributedRun: A handle on the submitted run.
        """
        # Imported here rather than at module scope so that the consume path stays usable
        # without the distributed stack, and to keep `crane.distributed` free to import from
        # `crane.core`.
        from ..distributed.core.run import submit as submit_run

        return submit_run(
            ds=ds,
            target=self,
            backend=on,
            save_dir=None,
            needs_finalize=False,
            finalizer=finalizer,
            finalizer_batch_size=batch_size,
            finalizer_formatting=formatting,
        )
