"""Progress Monitor Module.

This module provides the :class:`ProgressMonitor` class, which is designed to track data
processing tasks. It serves as a centralized mechanism to monitor the state and progression
of workers.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import threading
from enum import Enum
from typing import Any, Iterable, TypeAlias, TypedDict

from .runners.base import WorkerProcessingStage, WorkerRole
from .utils import EMA, TimeWeightedEMA, clock

TimeReport: TypeAlias = dict[WorkerProcessingStage, float]


class ProgressReport(TypedDict):
    """Represents a progress report detailing worker's timing and sample information."""

    timestamp: float
    """The timestamp when the report was generated."""

    elapsed_time: float
    """The time elapsed since the last report, in seconds."""

    num_samples: int
    """The number of samples processed since the last report."""

    average_elapsed_time: TimeReport
    """The average time spent by the worker in each role (producer, processor, finalizer)."""

    total_elapsed_time: TimeReport
    """The total time spent by the worker in each role (producer, processor, finalizer)."""


class ShardState(Enum):
    """Enum representing the processing state of a dataset shard."""

    PENDING = 1
    """
    Indicates that the shard is pending and has not yet been assigned to a worker.
    """

    IN_PROGRESS = 2
    """
    Indicates that the shard is currently being processed by a worker.
    """

    COMPLETED = 3
    """
    Indicates that the shard has been fully processed by a worker.
    """

    CANCELED = 4
    """
    Indicates that the processing of the shard has been canceled before completion.
    """


class ProgressMonitor(object):
    """Monitors the state and progress of all workers.

    The :class:`ProgressMonitor` tracks the state of each shard, the role and status of each
    worker, and the number of samples processed, providing utilities for reporting progress and
    checking worker and shard statuses.
    """

    def __init__(
        self,
        num_shards: int,
        num_workers: int,
        queue: mp.Queue | None,
        item_size: int,
        update_interval: float = 0.1,
    ) -> None:
        """Initializes the :class:`ProgressMonitor`.

        Args:
            num_shards (int): The total number of shards to process.
            num_workers (int): The number of workers processing the data.
            queue (None | mp.Queue): Sample queue filled by producer workers.
            item_size (int): The number of samples contained within a queue item.
            update_interval (float): The update interval in seconds.
        """
        self._stopping = threading.Event()
        self._done = threading.Event()
        # track buffer queue
        self._queue = queue
        self._item_size = item_size
        self._ema_queue_size = TimeWeightedEMA(decay_rate=math.log(2) * 0.01)
        # track shards
        self._num_shards = num_shards
        self._shard_state = [ShardState.PENDING] * num_shards
        # track workers
        self._num_workers = num_workers
        self._alive = [False] * num_workers
        self._roles = [None] * num_workers
        self._shard = [None] * num_workers

        # capture smooth throughput
        self._last_report = {role: clock() for role in WorkerRole}
        self._smooth_dn = {role: EMA(smoothing=0.3) for role in WorkerRole}
        self._smooth_dt = {role: EMA(smoothing=0.3) for role in WorkerRole}

        # track worker metrics
        self._time_ema = [
            {
                stage: TimeWeightedEMA(math.log(2) / 1.0)  # 1s half-life time
                for stage in WorkerProcessingStage
            }
            for _ in range(num_workers)
        ]
        self._num_samples = [{role: 0 for role in WorkerRole} for _ in range(num_workers)]

        self._update_interval = update_interval
        self._monitor_cache: dict[str, Any] = {
            role: {"should_update": False, "acc_num_samples": 0, "acc_time_delta": 0.0}
            for role in WorkerRole
        }
        self._monitor_cache_lock = threading.Lock()
        self._monitor_thread = threading.Thread(target=self._monitor_thread_fn)
        self._monitor_thread.start()

    @property
    def num_shards(self) -> int:
        """Returns the total number of dataset shards to process.

        Returns:
            int: The total number of shards.
        """
        return self._num_shards

    @property
    def num_workers(self) -> int:
        """Returns the total number of workers.

        Returns:
            int: The total number of workers.
        """
        return self._num_workers

    @property
    def is_stopping(self) -> bool:
        """Checks if the process is in a stopping state.

        Returns:
            bool: True if the process is stopping, False otherwise.
        """
        return self._stopping.is_set()

    @property
    def is_done(self) -> bool:
        """Checks if the process is done.

        Returns:
            bool: True if the process is done, False otherwise.
        """
        return self._done.is_set()

    @property
    def any_pending_shards(self) -> bool:
        """Checks if there are any pending shards.

        Returns:
            bool: True if there are shards pending, False otherwise.
        """
        return ShardState.PENDING in set(self._shard_state)

    @property
    def pending_shards(self) -> set[int]:
        """Returns a set of shard IDs that are pending.

        Returns:
            set[int]: A set of shard IDs in the :code:`PENDING` state.
        """
        return {i for i, state in enumerate(self._shard_state) if state is ShardState.PENDING}

    @property
    def completed_shards(self) -> set[int]:
        """Returns a set of shard IDs that have been completed.

        Returns:
            set[int]: A set of shard IDs in the :code:`COMPLETED` state.
        """
        return {i for i, state in enumerate(self._shard_state) if state is ShardState.COMPLETED}

    @property
    def any_worker_alive(self) -> bool:
        """Checks if any workers are currently alive.

        Returns:
            bool: True if any workers are alive, False otherwise.
        """
        return any(self._alive)

    @property
    def alive_workers(self) -> set[int]:
        """Returns a set of ranks of all workers that are currently alive.

        Returns:
            set[int]: A set of worker ranks that are alive.
        """
        return {i for i, alive in enumerate(self._alive) if alive}

    @property
    def idle_workers(self) -> set[int]:
        """Returns a set of ranks of workers that are currently idle.

        Returns:
            set[int]: A set of worker ranks that are idle (i.e., have no assigned role).
        """
        return {i for i, role in enumerate(self._roles) if role is None}

    @property
    def busy_workers(self) -> set[int]:
        """Returns a set of ranks of workers that are currently busy.

        Returns:
            set[int]: A set of worker ranks that are busy.
        """
        return {i for i, role in enumerate(self._roles) if role is not None}

    @property
    def num_buffered_samples(self) -> int:
        """Returns the number of samples currently buffered in the queue awaiting processing.

        This property calculates the total number of samples that are waiting in the queue by
        multiplying the number of items in the queue by the item size (the number of samples
        contained in each item).

        Returns:
            int: The total number of samples currently buffered in the queue.
        """
        return int(self._ema_queue_size.value) if self._queue is not None else 0

    @property
    def num_processed_samples(self) -> int:
        """Returns the total number of samples processed by all workers.

        The count includes samples processed by both processors and consumers.
        It excludes the samples processed by producers as these samples are not
        finalized yet and would be counted twice, once by the producer and and
        once by the consumer.

        Returns:
            int: The total number of processed samples.
        """
        return sum(
            nums[WorkerRole.STANDALONE] + nums[WorkerRole.CONSUMER] for nums in self._num_samples
        )

    @property
    def samples_per_second(self) -> float:
        """Calculate the throughput as the ratio of processed samples to time.

        This method computes the throughput by dividing the smoothed number of
        processed samples by the smoothed elapsed time between the current and
        previous progress reports.

        Returns:
            float: The calculated throughput, representing the number of processed
            samples per unit of time.
        """
        return (
            self._smooth_dn[WorkerRole.STANDALONE].value
            / max(self._smooth_dt[WorkerRole.STANDALONE].value, 1e-5)
        ) + (
            self._smooth_dn[WorkerRole.CONSUMER].value
            / max(self._smooth_dt[WorkerRole.CONSUMER].value, 1e-5)
        )

    def get_worker_shard(self, rank: int) -> None | int:
        """Retrieves the shard currently assigned to the worker.

        Args:
            rank (int): The rank of the worker whose assigned shard is to be retrieved.

        Returns:
            None | int: The shard ID assigned to the worker, or None if no shard is currently
            assigned to the worker.
        """
        return self._shard[rank]

    def get_worker_role(self, rank: int) -> None | WorkerRole:
        """Returns the current role of a worker.

        Args:
            rank (int): The worker's rank.

        Returns:
            WorkerRole | None: The role of the worker or None if the worker has no assigned role.
        """
        return self._roles[rank]

    def get_workers_with_role(self, role: WorkerRole) -> set[int]:
        """Returns a set of ranks of workers assigned a specific role.

        Args:
            role (WorkerRole): The role to filter workers by.

        Returns:
            set[int]: A set of worker ranks that are assigned the specified role.
        """
        return {i for i, r in enumerate(self._roles) if r is role}

    def elapsed_time_averages(self, ranks: None | Iterable[int] = None) -> TimeReport:
        """Returns the average elapsed time for each stage of the worker pipeline.

        The times of measured by each worker are accumulated using a time-weighted
        exponential moving average with a half-life time of one second.

        Args:
            ranks (Iterable[int] | None): Optional list of worker ranks to calculate
                average times for. If None, all workers are included.

        Returns:
            TimeReport: A dictionary mapping worker processing stages to their
            respective average elapsed times.
        """
        ranks = ranks if ranks is not None else range(self.num_workers)

        elapsed_times = {}
        for stage in WorkerProcessingStage:
            # filter and sort ranks by timestamp
            def get_timestamp(r):
                return self._time_ema[r][stage].timestamp

            # get the valid ranks for the given stage and sort them
            valid_ranks = filter(get_timestamp, ranks)
            valid_ranks = sorted(valid_ranks, key=get_timestamp)

            global_stage_ema = TimeWeightedEMA(math.log(2) / 1.0)
            # accumulate the elapsed times of each worker into a
            # global time-weighted average
            for rank in valid_ranks:
                local_ema = self._time_ema[rank][stage]
                global_stage_ema.update(local_ema.timestamp, local_ema.value)

            elapsed_times[stage] = global_stage_ema.value

        return elapsed_times

    def _monitor_thread_fn(self) -> None:
        """Continuously update and monitor the processing state.

        This method runs in a separate thread at regular intervals defined by
        :code:`_update_interval`, performing two primary tasks:

        1. **Updating Monitoring State:**
           Iterates through worker roles to update the time-weighted exponential moving
           averages (EMA) of sample counts and time deltas, based on accumulated values in
           the monitoring cache. The cache is then reset for the next interval.

        2. **Tracking Queue State:**
           If a queue is available, it calculates the current queue size (scaled by
           :code:`_item_size`) and updates its EMA over time. This provides a smooth estimate
           of queue utilization.

        The method stops execution when the :code:`_done` event is set. It also handles
        exceptions caused by issues in queue connectivity.

        Raises:
            BrokenPipeError: If the queue connection is broken.
            ConnectionResetError: If the queue connection is reset.
        """
        try:
            while not self._done.wait(timeout=self._update_interval):
                with self._monitor_cache_lock:
                    for role in WorkerRole:
                        if self._monitor_cache[role]["should_update"]:
                            self._smooth_dn[role].update(
                                self._monitor_cache[role]["acc_num_samples"]
                            )
                            self._smooth_dt[role].update(
                                self._monitor_cache[role]["acc_time_delta"]
                            )

                        self._monitor_cache[role]["should_update"] = False
                        self._monitor_cache[role]["acc_num_samples"] = 0
                        self._monitor_cache[role]["acc_time_delta"] = 0.0

                if self._queue is not None:
                    # compute queue size and update the moving average
                    qsize = self._queue.qsize() * self._item_size
                    self._ema_queue_size.update(clock(), qsize)

        except (BrokenPipeError, ConnectionResetError):  # pragma: not covered
            # queue connection closed
            return

    def _report_progress(self, rank: int, report: ProgressReport) -> None:
        """Capture the progress report of a worker.

        This method updates the time-weighted averages of worker progress,
        the number of samples processed, and the time elapsed between reports.

        Args:
            rank (int): The rank of the worker reporting progress.
            report (ProgressReport): A dictionary containing the worker's report data, including
                timestamps, average processing times, and the number of samples processed.

        Raises:
            AssertionError: If the worker's role is None, indicating the worker is not assigned
                a valid role during progress reporting.
        """
        role = self.get_worker_role(rank)
        assert (
            role is not None
        ), f"Worker {rank} does not have an assigned role during progress reporting."

        ts = report["timestamp"]
        # update average time spend in each stage
        time_report = report["average_elapsed_time"]
        for stage, elapsed_time in time_report.items():
            self._time_ema[rank][stage].update(ts, elapsed_time)
        # update processed samples
        role = self._roles[rank]
        self._num_samples[rank][role] += report["num_samples"]
        # update smooth deltas
        with self._monitor_cache_lock:
            self._monitor_cache[role]["should_update"] = True
            self._monitor_cache[role]["acc_num_samples"] += report["num_samples"]
            self._monitor_cache[role]["acc_time_delta"] += ts - self._last_report[role]
        self._last_report[role] = ts

    def _mark_as_stopping(self) -> None:
        """Marks the process as stopping."""
        self._stopping.set()

    def _mark_as_done(self) -> None:
        """Marks the process as done."""
        self._done.set()
        # wait for the monitor thread to terminate
        self._monitor_thread.join()

    def _mark_shard_in_progress(self, rank: int, shard_id: int) -> None:
        """Marks a shard as being in progress, associating it with a worker.

        Args:
            rank (int): The rank of the worker processing the shard.
            shard_id (int): The identifier of the shard being processed.
        """
        self._shard_state[shard_id] = ShardState.IN_PROGRESS
        self._shard[rank] = shard_id

    def _mark_shard_completed(self, shard_id: int) -> None:
        """Marks a shard as completed once processing is finished.

        Args:
            shard_id (int): The identifier of the shard that is completed.
        """
        self._shard_state[shard_id] = ShardState.COMPLETED

    def _mark_shard_canceled(self, shard_id: int) -> None:
        """Marks a shard as canceled.

        Args:
            shard_id (int): The identifier of the shard that is canceled.
        """
        self._shard_state[shard_id] = ShardState.CANCELED

    def _mark_worker_ready(self, rank: int) -> None:
        """Marks a worker as ready to process tasks.

        Args:
            rank (int): The rank of the worker being marked as ready.
        """
        self._alive[rank] = True

    def _mark_worker_done(self, rank: int) -> None:
        """Marks a worker as done and no longer processing.

        Args:
            rank (int): The rank of the worker being marked as done.
        """
        assert self._roles[rank] is None
        self._alive[rank] = False

    def _mark_worker_idling(self, rank: int) -> None:
        """Marks a worker as idle, with no current role.

        Args:
            rank (int): The rank of the worker being marked as idle.
        """
        assert self._alive[rank]
        self._roles[rank] = None

    def _mark_worker_busy(self, rank: int, role: WorkerRole) -> None:
        """Marks a worker as busy with a specific role.

        Args:
            rank (int): The rank of the worker being marked as busy.
            role (WorkerRole): The role that the worker is performing.
        """
        assert self._alive[rank]
        self._roles[rank] = role

    def _mark_worker_completed(self, rank: int) -> None:
        """Marks a worker's current task as completed.

        Also marks the corresponding shard as completed.

        Args:
            rank (int): The rank of the worker who has completed.
        """
        shard_id = self._shard[rank]
        if shard_id is not None:
            self._mark_shard_completed(shard_id)
            self._shard[rank] = None

    def _mark_worker_canceled(self, rank: int) -> None:
        """Marks a worker's current task as completed.

        Also marks the corresponding shard as canceled.

        Args:
            rank (int): The rank of the worker who has canceled their task.
        """
        shard_id = self._shard[rank]
        if shard_id is not None:
            self._mark_shard_canceled(shard_id)
            self._shard[rank] = None
