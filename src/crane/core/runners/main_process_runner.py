"""Main Process Runner Module.

This module provides the :class:`MainProcessRunner` class, which implements a data processing
pipeline that runs entirely in the main thread. It processes data in batches, applies a
user-defined function to each sample, and reports progress through an optional progress
bar. The runner manages initialization and finalization tasks and handles progress reporting
through callbacks.
"""

import logging
from typing import Any, Callable, TypeAlias

from datasets.iterable_dataset import FormattingConfig, IterableDataset

from ..callbacks.base import CallbackManager
from ..iterables import TimedExamplesIterable
from ..monitor import ProgressMonitor, ProgressReport
from ..utils import clock
from ..worker import reset_worker_info, set_worker_info
from .base import BaseRunner, WorkerProcessingStage, WorkerRole

logger = logging.getLogger(__name__)

Stages: TypeAlias = WorkerProcessingStage


class MainProcessRunner(BaseRunner):
    """Runs data processing in the main thread.

    This class implements the data processing pipeline in the main thread, handling data
    consumption and applying a function to each sample. It supports progress reporting
    with optional progress bar display and handles initialization and finalization tasks.
    """

    def __init__(
        self,
        batch_size: int,
        env_init: Callable[[], Any],
        env_finalize: Callable[[], Any],
        progress_report_interval: float,
        callback: CallbackManager,
    ) -> None:
        """Initialize the MainProcessRunner.

        Args:
            batch_size (int): The number of samples to process in each batch.
            env_init (Callable[[], Any]): Function to initialize the processing environment.
            env_finalize (Callable[[], Any]): Function to finalize the processing environment.
            progress_report_interval (float): The time interval, in seconds, between sending
                progress updates.
            callback (CallbackManager): A callback manager that will be invoked at various points
                during the data processing lifecycle.
        """
        self._batch_size = batch_size
        self._env_init = env_init
        self._env_finalize = env_finalize
        self._report_interval = progress_report_interval
        self._callback = callback

    def run(
        self,
        ds: IterableDataset,
        finalizer: Callable[[Any], Any],
        batch_size: None | int = None,
        formatting: None | str = None,
    ) -> None:
        """Execute data processing on the dataset.

        This method prepares the dataset, initializes the processing environment, and processes
        the dataset in batches. It reports progress to a tracker and updates a progress bar
        if not disabled. The method handles initialization and finalization of the processing
        environment and manages progress reporting throughout the processing.

        Args:
            ds (IterableDataset): The dataset to process.
            finalizer (Callable[[Any], Any]): The function to apply to each sample or batch
                of samples in the dataset as the final processing step.
            batch_size (None | int): The size of each batch to process. If :code:`None`,
                process samples individually. Only affects the finalizer. Defaults to None.
            formatting (None | str): The data format in which samples or batches are provided
                to the finalizer function.
        """
        logger.info("Preparing to run data processing on dataset.")

        # set the worker info
        set_worker_info(rank=0, num_workers=1, seed=None)

        num_shards = ds.n_shards
        # prepare the dataset
        ds = ds._prepare_ex_iterable_for_iteration(batch_size=self._batch_size)
        logger.info(f"Dataset prepared with {num_shards} shards.")

        # create the progress monitor
        monitor = ProgressMonitor(num_shards, 1, None, 0)

        try:
            # call start callback
            self._callback.on_start(monitor, ds)

            # initialize the consumer
            self._env_init()
            logger.info("Initialization complete.")

            monitor._mark_worker_ready(0)
            monitor._mark_worker_idling(0)

            num_samples = 0
            last_report = clock()

            for shard_id in range(num_shards):
                try:
                    monitor._mark_shard_in_progress(0, shard_id)
                    monitor._mark_worker_busy(0, WorkerRole.STANDALONE)
                    self._callback.on_shard_in_progress(monitor, shard_id)

                    # get the dataset shard and time it
                    shard = ds.shard_data_sources(num_shards, shard_id)
                    stream = TimedExamplesIterable(shard, smoothing=0.1)

                    # apply the function
                    work_iterator = (
                        IterableDataset(
                            ex_iterable=stream,
                            formatting=(
                                None
                                if formatting is None
                                else FormattingConfig(format_type=formatting)
                            ),
                        )
                        .map(
                            lambda data: (finalizer(data) or data),
                            batched=batch_size is not None,
                            batch_size=batch_size or 1,
                        )
                        ._ex_iterable
                    )
                    work_iterator = TimedExamplesIterable(work_iterator, smoothing=0.1)

                    # main worker loop
                    for _ in work_iterator:
                        num_samples += 1

                        now = clock()
                        if now - last_report > self._report_interval:
                            report = ProgressReport(
                                timestamp=now,
                                num_samples=num_samples,
                                elapsed_time=now - last_report,
                                average_elapsed_time={
                                    Stages.STREAM.value: stream.smooth_time(),
                                    Stages.TRANSFORM.value: stream.smooth_time(),
                                    Stages.FINALIZE.value: work_iterator.smooth_time(),
                                },
                                total_elapsed_time={
                                    Stages.STREAM.value: stream.total_time(),
                                    Stages.TRANSFORM.value: stream.total_time(),
                                    Stages.FINALIZE.value: work_iterator.total_time(),
                                },
                            )
                            monitor._report_progress(0, report)
                            # reset
                            num_samples, last_report = 0, now

                    logger.info(f"Finished processing shard {shard_id}.")

                    monitor._mark_worker_completed(0)

                except KeyboardInterrupt:
                    # handle exception
                    monitor._mark_worker_canceled(0)
                    self._callback.on_shard_canceled(monitor, shard_id)
                    raise

                except Exception as e:
                    # handle exception
                    monitor._mark_worker_canceled(0)
                    self._callback.on_shard_canceled(monitor, shard_id)
                    # log
                    logger.error(f"Unexpected error during processing: {str(e)}.", exc_info=True)

                else:
                    # call shard complete when no error was detected
                    self._callback.on_shard_completed(monitor, shard_id)

        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user.")
            monitor._mark_worker_canceled(0)

        finally:
            # report worker done
            logger.info("Processing completed. Finalizing worker.")

            if num_samples > 0:
                now = clock()
                report = ProgressReport(
                    timestamp=now,
                    num_samples=num_samples,
                    elapsed_time=now - last_report,
                    average_elapsed_time={stage: 0 for stage in Stages},
                    total_elapsed_time={stage: 0 for stage in Stages},
                )
                monitor._report_progress(0, report)

            try:
                # finalize the consumer
                self._env_finalize()
                logger.info("Worker finalized successfully.")
            except Exception as e:  # pragma: not covered
                logger.error(f"Error during worker finalization: {e}", exc_info=True)

            # stop worker
            monitor._mark_worker_idling(0)
            monitor._mark_worker_done(0)
            # reset the worker info
            reset_worker_info()
            # stopping
            monitor._mark_as_stopping()
            self._callback.on_stopping(monitor)
            # done
            monitor._mark_as_done()
            self._callback.on_done(monitor)
