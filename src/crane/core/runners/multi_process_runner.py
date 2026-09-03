"""Dynamic Multiprocessing Runner Module.

This module implements the :class:`DynamicMultiprocessingRunner` class, which manages and runs
multiple worker processes for parallel data processing. The runner dynamically assigns tasks
to workers, facilitating efficient data processing in two distinct stages:

1. **Single-Shard Single-Worker**: Each worker processes one dataset shard at a time.
2. **Single-Shard Multiple-Workers**: Multiple workers process the same shard, with distinct roles
   for producers (feeding a queue) and consumers (processing from the queue).

The runner optimizes resource usage and performance by adapting to the workload dynamically,
ensuring effective parallel processing throughout the data lifecycle.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.connection  # noqa: F401
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from functools import partial
from queue import Empty, Full
from typing import Any, Callable, TypeAlias

import dill
import orjson
from datasets import IterableDataset
from datasets.iterable_dataset import (
    FilteredExamplesIterable,
    FormattingConfig,
    MappedExamplesIterable,
    RebatchedArrowExamplesIterable,
    SelectColumnsIterable,
    _BaseExamplesIterable,
    identity_func,
)

from ..callbacks.base import CallbackManager
from ..iterables import (
    ExamplesIterablePipeline,
    FastRebatchedArrowExamplesIterable,
    QueueExamplesIterable,
    StoppableExamplesIterable,
    TimedExamplesIterable,
)
from ..monitor import ProgressMonitor, ProgressReport
from ..utils import clock
from ..worker import set_worker_info
from .base import (
    BaseRunner,
    FailurePolicy,
    ShardFailure,
    ShardProcessingError,
    WorkerProcessingStage,
    WorkerRole,
)

# shorthands and helper type aliases
Stages: TypeAlias = WorkerProcessingStage

# How long to keep trying to hand a command to a worker before giving up on that
# particular command. The race this absorbs is the queue feeder thread being momentarily
# behind, which resolves in milliseconds, so keep it short: commands are also sent from
# the balancer's hot path and a long block there would stall coordination.
_COMMAND_SEND_TIMEOUT: float = 1.0
_COMMAND_SEND_RETRY_INTERVAL: float = 0.01

# How long a consumer keeps waiting on a queue that is delivering nothing at all. Only a
# backstop against a lost close signal - a consumer normally leaves the moment the stream
# is marked closed - so reaching it means something went wrong rather than that the run
# ended.
_QUEUE_GET_TIMEOUT: float = 30.0

# The lazy processing steps that `_prepare_dataset` separates from the data source.
# `FormattedExamplesIterable` was introduced in newer `datasets` releases (it is inserted
# into the iterable chain by `.map`); guard the import so crane keeps working on versions
# where it does not exist yet.
_SEPARABLE_EX_ITERABLES: tuple[type, ...] = (
    MappedExamplesIterable,
    FilteredExamplesIterable,
    RebatchedArrowExamplesIterable,
)
try:
    from datasets.iterable_dataset import FormattedExamplesIterable as _FormattedExamplesIterable

    _SEPARABLE_EX_ITERABLES += (_FormattedExamplesIterable,)
except ImportError:
    pass

# Column projections are separable, but only conditionally. `.map(remove_columns=...)`
# inserts a `SelectColumnsIterable` mid-chain, and if the walk stops there every step
# below it stays with the data source: those steps then run in the producers, and
# sharding a source that still contains an arrow-formatted map fails outright with
# "doesn't implement iter_arrow()".
#
# Separating a projection unconditionally is not right either. One sitting directly on
# the source is pure projection, and keeping it in the producer means fewer columns
# travel through the queue. So a projection is separated only when there is something
# separable beneath it, which is exactly the case where leaving it behind would strand
# real work in the producers.
_PROJECTION_EX_ITERABLES: tuple[type, ...] = (SelectColumnsIterable,)


def _has_separable_below(ex_iterable: _BaseExamplesIterable) -> bool:
    """Whether any step beneath `ex_iterable` is itself separable processing."""
    node = getattr(ex_iterable, "ex_iterable", None)
    while node is not None:
        if isinstance(node, _SEPARABLE_EX_ITERABLES):
            return True
        node = getattr(node, "ex_iterable", None)
    return False


def _is_separable(ex_iterable: _BaseExamplesIterable) -> bool:
    """Whether a step should be moved off the data source and into the workers.

    Processing steps always are. A column projection is only worth separating when real
    work sits beneath it; see `_PROJECTION_EX_ITERABLES`.
    """
    if isinstance(ex_iterable, _SEPARABLE_EX_ITERABLES):
        return True
    if isinstance(ex_iterable, _PROJECTION_EX_ITERABLES):
        return _has_separable_below(ex_iterable)
    return False


class _CountedQueue:
    """An `mp.Queue` that can report its own length on every platform.

    The runner used a `mp.Manager().Queue()` from its first commit. That is expensive: a
    manager queue is a proxy served by a separate process, so every batch of data is
    pickled, shipped over a socket, held in that process, then pickled again on its way
    to the consumer - and the progress monitor polls `qsize()` every 10 ms, which on a
    proxy is another socket round-trip a hundred times a second, competing with the data
    it is measuring.

    What stood in the way of simply swapping in a plain `mp.Queue` is that the balancer
    needs the queue's depth, and `mp.Queue.qsize()` is backed by `sem_getvalue()`, which
    macOS does not implement.

    Counting puts and gets ourselves keeps a plain `mp.Queue` - one pickle, one pipe, no
    intermediary - and gives an accurate length on macOS too. The count trails reality
    slightly, since `mp.Queue` hands items to a feeder thread, but it drives an EMA for
    the producer/consumer balancer, which wants a trend rather than an exact depth.

    Not picklable, by design: like the `mp.Queue` it wraps, it reaches workers through
    `fork`, never through a pipe.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue = mp.Queue(maxsize=maxsize)
        self._size = mp.Value("i", 0)

    def _bump(self, delta: int) -> None:
        with self._size.get_lock():
            self._size.value += delta

    def put(self, item: Any, block: bool = True, timeout: None | float = None) -> None:
        """Put an item on the queue."""
        self._queue.put(item, block, timeout)
        self._bump(1)

    def put_nowait(self, item: Any) -> None:
        """Put an item on the queue without blocking."""
        self.put(item, block=False)

    def get(self, block: bool = True, timeout: None | float = None) -> Any:
        """Take an item off the queue."""
        item = self._queue.get(block, timeout)
        self._bump(-1)
        return item

    def get_nowait(self) -> Any:
        """Take an item off the queue without blocking."""
        return self.get(block=False)

    def qsize(self) -> int:
        """Approximate number of items currently queued."""
        return max(0, self._size.value)

    def empty(self) -> bool:
        """Whether the queue is currently empty."""
        return self.qsize() == 0


class Source(Enum):
    """Where a role takes its data from."""

    SHARD = "shard"
    """A shard of the source dataset, named by the command."""

    QUEUE = "queue"
    """The queue shared with the producers."""

    CURRENT = "current"
    """Whatever stream the worker is already holding; a role that never brings its own."""


@dataclass(frozen=True)
class WorkerContext:
    """How a worker processes while it holds one role.

    Built once per worker, before it starts, and never sent anywhere. A worker holds one
    of these per role and swaps between them on command, so the only thing that has to
    travel at runtime is which role to take; see :class:`Command`.
    """

    role: WorkerRole
    """The role this context implements."""

    transform: Callable[[_BaseExamplesIterable], _BaseExamplesIterable]
    """Wraps the data stream and applies the workload to it."""

    finalizer: Callable[[Any], Any]
    """Applied to each sample or batch after the transform."""

    finalizer_batch_size: None | int
    """Batch size for :attr:`finalizer`, or None to pass individual samples."""

    finalizer_formatting: None | FormattingConfig
    """The data format :attr:`finalizer` expects."""

    source: Source
    """Where this role reads from."""

    def __str__(self) -> str:
        """Returns a string representation of the context, including the worker role."""
        return f"WorkerContext(role={self.role.name}, source={self.source.name})"


@dataclass(frozen=True)
class WorkerSetup:
    """Everything a worker needs that is fixed for the whole run.

    Handed over once, at construction. Under `fork` the worker inherits it through
    copy-on-write and nothing is serialised at all; under `spawn` it is dill-pickled once
    in :meth:`DynamicMultiprocessingRunner.run` and the same bytes go to every worker, so
    the cost is one dump per run and one load per worker rather than anything per command.

    Dill rather than pickle because a transform is routinely a closure, a lambda or a
    partial over a local, which the standard pickler cannot take. That is also why the
    queue is *not* in here: multiprocessing objects reach a worker through process
    creation, never through a serialised blob, so they are passed to
    :class:`Worker` separately.
    """

    transform: Callable[[_BaseExamplesIterable], _BaseExamplesIterable]
    """The workload, shared by the standalone and consumer roles."""

    finalizer: Callable[[Any], Any]
    """The write step, shared by the standalone and consumer roles."""

    finalizer_batch_size: None | int
    """Batch size for :attr:`finalizer`."""

    finalizer_formatting: None | FormattingConfig
    """The data format :attr:`finalizer` expects."""

    data_source: _BaseExamplesIterable
    """The dataset the standalone role takes its shards from."""

    num_shards: int
    """How many shards :attr:`data_source` is divided into."""


class CommandType(Enum):
    """What the controller is telling a worker to do.

    The counterpart of :class:`MessageType`, which travels the other way: workers report,
    the controller commands.
    """

    ASSIGN = 1
    """Take a new stream and start on it. Only an idle worker accepts one."""

    SWITCH = 2
    """Change role but keep the current stream. Only a busy worker accepts one."""

    STOP = 3
    """Finish and shut down."""


@dataclass(frozen=True)
class Command:
    """An instruction from the controller to one worker. The whole wire payload.

    Small enough to travel as a plain object on the command queue, which pickles it with
    the standard pickler - so, unlike the pipeline it replaces, it cannot drag a closure
    along by accident.
    """

    type: CommandType
    """What to do."""

    role: None | WorkerRole = None
    """Which of the worker's contexts to make active. None only for :attr:`CommandType.STOP`."""

    shard_id: None | int = None
    """Which shard to take, for an :attr:`CommandType.ASSIGN` whose role reads shards."""

    def __str__(self) -> str:
        """Returns a string representation of the command."""
        role = self.role.name if self.role is not None else "-"
        return f"Command({self.type.name}, role={role}, shard={self.shard_id})"


class MessageType(Enum):
    """Enum representing the various worker message types."""

    READY = 1
    """
    Indicates that a worker is ready to start processing or receive tasks.
    """

    DONE = 2
    """
    Indicates that a worker has completed processing and terminates soon.
    """

    EXCEPTION = 3
    """
    Indicates that an exception occurred during worker processing.
    Used to signal errors or abnormal terminations in task execution.
    """

    CTX_REQUEST = 4
    """
    Requests a new context for the worker.
    """

    CTX_STARTED = 5
    """
    Signals that a worker has started processing a given context.
    """

    CTX_REPORT = 6
    """
    Provides status updates and progress reports for a worker's current context.
    """

    CTX_SWITCH = 7
    """
    Indicates that the worker is requesting to switch to a different context.
    """

    CTX_COMPLETE = 8
    """
    Signals that the worker has successfully completed its current context.
    """

    CTX_CANCELED = 9
    """
    Indicates that the current context has been canceled before completion.
    """


class Worker(mp.Process):
    """A worker process for parallel data processing.

    This class extends :code:`mp.Process` to handle data processing in a separate
    process, managing context and processing stages for data.
    """

    def __init__(
        self,
        rank: int,
        num_workers: int,
        msg_queue: mp.Queue,
        progress_report_interval: float,
        worker_init: Callable[[], Any],
        worker_finalize: Callable[[], Any],
        setup: WorkerSetup | bytes,
        queue_stream: None | _BaseExamplesIterable = None,
        queue_put: None | Callable[[Any], None] = None,
        prefetch: int = 1,
    ) -> None:
        """Initialize a worker process for parallel data processing.

        Args:
            rank (int): The rank or identifier for the worker, typically representing
                the worker's position or role in the set of workers.
            num_workers (int): The total number of workers involved in the data
                processing pipeline.
            msg_queue (mp.Queue): A queue object used to send messages to the manager process.
            progress_report_interval (float): The time interval, in seconds, between sending
                progress updates.
            worker_init (Callable[[], Any]): A callable function to initialize the worker's
                state before processing begins.
            worker_finalize (Callable[[], Any]): A callable function to finalize the worker's
                state after processing is complete.
            setup (WorkerSetup | bytes): Everything fixed for the run - the workload, the
                write step and the dataset to take shards from. Passed as an object where
                the worker inherits memory, or as dill bytes where it does not; see
                :class:`WorkerSetup`.
            queue_stream (None | _BaseExamplesIterable): The shared queue, as the consumer
                role reads it.
            queue_put (None | Callable[[Any], None]): The shared queue's put, as the
                producer role writes it.
            prefetch (int): How many samples a producer batches into one queue item.
        """
        super(Worker, self).__init__(daemon=True)

        self._rank = rank
        self._num_workers = num_workers
        # connections
        self._msg_queue = msg_queue
        self._command_queue = mp.Queue(maxsize=1)
        self._recv_resp_conn, self._send_resp_conn = mp.Pipe(duplex=False)
        # rate limit for progress updates
        self._progress_report_interval = progress_report_interval
        # worker initializer and finalizer
        self._worker_init = worker_init
        self._worker_finalize = worker_finalize
        # Resolved in `run`, in the child, because under `spawn` it arrives as bytes.
        self._setup = setup
        self._queue_stream = queue_stream
        self._queue_put = queue_put
        self._prefetch = prefetch
        # One context per role, built in `_build_contexts` once the setup is resolved.
        # The active one is always one of these three; commands only ever swap between
        # them, which is why nothing about the pipeline has to be sent.
        self._standalone_ctx: None | WorkerContext = None
        self._producer_ctx: None | WorkerContext = None
        self._consumer_ctx: None | WorkerContext = None
        self._active_ctx: None | WorkerContext = None
        # The stream the active context is working through, and whether we were told to
        # finish. A `SWITCH` keeps the stream; an `ASSIGN` replaces it.
        self._stream: None | _BaseExamplesIterable = None
        self._stop = False

        # create logger
        self._logger = logging.getLogger(f"{type(self).__module__}.{type(self).__qualname__}")

        # Log initialization details
        self._logger.debug(
            f"Created worker with rank {self._rank} of {self._num_workers} total workers."
        )

    def _build_contexts(self) -> None:
        """Build one context per role, from the setup and the shared queue.

        The standalone and consumer roles run the same pipeline and differ only in where
        they read from; they are still built separately, because that is a fact about the
        current runner rather than a rule, and three named roles read better than a
        consumer borrowing the standalone's pipeline.

        The producer context is assembled here rather than shipped, because everything in
        it is either a constant or the shared queue, both of which the worker already has.
        """
        if isinstance(self._setup, bytes):
            self._setup = dill.loads(self._setup)

        setup: WorkerSetup = self._setup
        self._standalone_ctx = WorkerContext(
            role=WorkerRole.STANDALONE,
            transform=setup.transform,
            finalizer=setup.finalizer,
            finalizer_batch_size=setup.finalizer_batch_size,
            finalizer_formatting=setup.finalizer_formatting,
            source=Source.SHARD,
        )
        self._consumer_ctx = WorkerContext(
            role=WorkerRole.CONSUMER,
            transform=setup.transform,
            finalizer=setup.finalizer,
            finalizer_batch_size=setup.finalizer_batch_size,
            finalizer_formatting=setup.finalizer_formatting,
            source=Source.QUEUE,
        )
        self._producer_ctx = WorkerContext(
            role=WorkerRole.PRODUCER,
            transform=QueueExamplesIterable.prepare_ex_iterable,
            finalizer=self._queue_put,
            finalizer_batch_size=self._prefetch,
            finalizer_formatting=FormattingConfig(format_type="arrow"),
            source=Source.CURRENT,
        )

    def _context_for(self, role: WorkerRole) -> WorkerContext:
        """The context implementing a role."""
        return {
            WorkerRole.STANDALONE: self._standalone_ctx,
            WorkerRole.PRODUCER: self._producer_ctx,
            WorkerRole.CONSUMER: self._consumer_ctx,
        }[role]

    def close(self) -> None:
        """Close the Process object.

        This method releases resources held by the Process object.  It is
        an error to call this method if the child process is still running.
        """
        # close connections and queues
        self._recv_resp_conn.close()
        self._send_resp_conn.close()
        self._command_queue.close()
        # close worker process
        super(Worker, self).close()

    def send_command(self, command: Command, blocking: bool = True) -> bool:
        """Send a command to the worker.

        Args:
            command (Command): The command to send.
            blocking (bool): Whether to block and wait for the worker's answer.

        Returns:
            bool: Whether the worker accepted the command. Always True when not blocking.
            A refusal is meaningful: a worker refuses a `SWITCH` while it is idle and an
            `ASSIGN` while it is busy, which is how the controller learns that a role
            change did not take and it should try another rank.
        """
        # clear connection buffer
        while self._recv_resp_conn.poll():
            self._recv_resp_conn.recv()

        self._send_command(command)

        if blocking:
            # wait for feedback from worker
            accepted = self._recv_resp_conn.recv()
            # log
            accept_str = "accepted" if accepted else "refused"
            self._logger.debug(f"Sent {command} to worker {self._rank}, worker {accept_str}.")
            # return
            return accepted

        else:
            self._logger.debug(f"Sent {command} to worker {self._rank} in non-blocking mode.")
            return True

    def _send_command(self, command: Command, timeout: float = _COMMAND_SEND_TIMEOUT) -> bool:
        """Replace whatever command is queued for the worker with `command`.

        The queue holds at most one pending command, so a stale one is dropped first.
        Draining and then calling `put_nowait` is not enough on its own: a
        `multiprocessing.Queue` is fed by a background thread, so a successful `get()`
        does not free the slot synchronously and an item already in flight can arrive
        between the two calls. `put_nowait` then raises `Full`, which used to escape into
        the controller's message loop and take it down - leaving the workers waiting for a
        context that never came, so they never joined and the run hung.

        Retry within a deadline instead, and treat exhaustion as a dropped command rather
        than a fatal error: the balancer sends these continuously, so the next one will
        carry the same information.

        Returns:
            bool: Whether the command reached the worker's queue.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._command_queue.get_nowait()
            except Empty:
                pass

            try:
                self._command_queue.put_nowait(command)
                return True
            except Full:
                if time.monotonic() >= deadline:
                    self._logger.warning(
                        f"Could not hand {command} to worker {self._rank} within "
                        f"{timeout}s; the worker is not consuming its command queue. "
                        f"Dropping it."
                    )
                    return False
                time.sleep(_COMMAND_SEND_RETRY_INTERVAL)

    def _send_msg(self, msg_type: MessageType, payload: None | Any = None) -> None:
        """Send a message from the worker to the manager process.

        This function serializes and sends a message to the manager process through the
        specified communication connection. The message includes the worker's rank,
        the type of message, and an optional payload.

        Args:
            msg_type (MessageType): The type of message to send, indicating the nature
                of the update or communication.
            payload (None | Any, optional): Additional data or context to be sent along with
                the message. Can be any serializable object. Default is `None`.
        """
        msg = {"rank": self._rank, "type": msg_type.value, "payload": payload}

        msg = orjson.dumps(msg)
        self._msg_queue.put(msg)

    def _request_work(self) -> bool:
        """Ask the controller for something to do, and wait until it answers.

        Returns:
            bool: Whether the worker has been instructed to stop.
        """
        self._logger.debug("Requesting work from main process.")
        self._send_msg(MessageType.CTX_REQUEST)

        while True:
            command = self._recv_command(blocking=True, timeout=1.0)
            if command is None:
                self._logger.debug("Waiting for a command...")
                continue

            if self._accept(command, busy=False):
                return self._stop

            # An idle worker has no stream to keep, so a role switch is meaningless here.
            # Refusing tells the controller to try another rank; ask again for real work.
            self._send_resp_conn.send(False)
            self._send_msg(MessageType.CTX_REQUEST)

    def _recv_command(self, blocking: bool, timeout: float = 1.0) -> Command | None:
        """Take the next command off the command queue, if there is one.

        Args:
            blocking (bool): Whether to block while waiting for a command.
            timeout (float, optional): Maximum time to wait when blocking. Defaults to
                1.0 seconds.

        Returns:
            Command | None: The command, or None if none arrived in time.
        """
        try:
            command = self._command_queue.get(block=blocking, timeout=timeout)
            self._logger.debug(f"Received {command}.")
            return command
        except Empty:
            return None

    def _accept(self, command: Command, busy: bool) -> bool:
        """Apply a command if it makes sense in the worker's current state.

        `ASSIGN` brings a new stream, so only an idle worker can take one; `SWITCH` keeps
        the stream the worker is working through, so only a busy worker can. Rejecting the
        mismatch is what the controller reads as "that rank was not available".

        Args:
            command (Command): The command to consider.
            busy (bool): Whether the worker is part-way through a stream.

        Returns:
            bool: Whether the command was applied.
        """
        if command.type is CommandType.STOP:
            self._stop = True
            self._send_resp_conn.send(True)
            self._logger.debug("Applied stop command.")
            return True

        if (command.type is CommandType.ASSIGN) is busy:
            return False

        self._active_ctx = self._context_for(command.role)
        if command.type is CommandType.ASSIGN:
            # Only an assign brings a stream. A switch changes how the worker processes
            # and leaves `self._stream` alone, which is what lets a producer carry on
            # with the shard it was promoted from - and lets it resume that same shard if
            # it is demoted back to standalone, even though the standalone role otherwise
            # reads shards.
            if self._active_ctx.source is Source.SHARD:
                self._stream = self._setup.data_source.shard_data_sources(
                    self._setup.num_shards, command.shard_id
                )
            elif self._active_ctx.source is Source.QUEUE:
                self._stream = self._queue_stream

        self._send_resp_conn.send(True)
        self._logger.debug(f"Applied {self._active_ctx}.")
        return True

    def run(self) -> None:
        """Start the worker process.

        Continuously request new contexts, process data with the current context, and
        apply new contexts as needed until instructed to stop.
        """
        # TODO: set seed

        # set worker info
        set_worker_info(
            rank=self._rank,
            num_workers=self._num_workers,
            seed=None,
        )

        try:
            # Build the role contexts before announcing readiness: under `spawn` this is
            # where the setup blob is unpickled, and it must happen in the child.
            self._build_contexts()
            # initialize worker
            self._worker_init()
            self._send_msg(MessageType.READY)
            self._logger.info("Initialization complete.")

            done = False
            # request a processing context
            while (not done) and (not self._request_work()):
                stream_exhausted = False
                # create the stoppable data stream iterator here to avoid
                # resetting it when a new context is received during processing
                data_stream = TimedExamplesIterable(self._stream, smoothing=0.1)
                stoppable_stream = StoppableExamplesIterable(data_stream)

                # exhaust producer
                while not stream_exhausted:
                    self._send_msg(MessageType.CTX_STARTED, payload=self._active_ctx.role.value)
                    self._logger.debug(f"Starting processing of {self._active_ctx}.")

                    num_samples = 0
                    last_report = clock()

                    stoppable_stream.resume()

                    try:
                        # apply the transformation function to the data stream
                        transformed_stream = self._active_ctx.transform(stoppable_stream)
                        transformed_stream = TimedExamplesIterable(
                            transformed_stream, smoothing=0.1
                        )
                        # apply the finalizer to the transformed stream
                        work_iterator = (
                            IterableDataset(
                                ex_iterable=transformed_stream,
                                formatting=self._active_ctx.finalizer_formatting,
                            )
                            .map(
                                lambda data: (self._active_ctx.finalizer(data) or data),
                                batched=self._active_ctx.finalizer_batch_size is not None,
                                batch_size=self._active_ctx.finalizer_batch_size,
                            )
                            ._ex_iterable
                        )
                        work_iterator = TimedExamplesIterable(work_iterator, smoothing=0.1)

                        # replace all rebatch operations with fast rebatch
                        # this especially replaces the rebatch operation that was added
                        # by the .map call earlier
                        work_iterator = FastRebatchedArrowExamplesIterable.replace_rebatch(
                            work_iterator
                        )

                        # prepare the iterator for iteration
                        work_iterator._init_state_dict()
                        # create the python iterable that executes the workload
                        # dynamically use the pyarrow iterable to avoid unnecessary conversion
                        iter_arrow = (self._active_ctx.finalizer_formatting is not None) and (
                            self._active_ctx.finalizer_formatting.format_type == "arrow"
                        )
                        it = work_iterator.iter_arrow() if iter_arrow else iter(work_iterator)

                        def _report_progress(now: float, num_samples: int, last_report: float):
                            payload = ProgressReport(
                                timestamp=now,
                                elapsed_time=now - last_report,
                                num_samples=num_samples,
                                average_elapsed_time={
                                    Stages.STREAM.value: data_stream.smooth_time(),
                                    Stages.TRANSFORM.value: transformed_stream.smooth_time(),
                                    Stages.FINALIZE.value: work_iterator.smooth_time(),
                                },
                                total_elapsed_time={
                                    Stages.STREAM.value: data_stream.total_time(),
                                    Stages.TRANSFORM.value: transformed_stream.total_time(),
                                    Stages.FINALIZE.value: work_iterator.total_time(),
                                },
                            )
                            self._send_msg(MessageType.CTX_REPORT, payload=payload)

                        # main worker loop
                        for _, data in it:
                            num_samples += data.num_rows if iter_arrow else 1

                            # check whether the controller wants this worker elsewhere
                            if (command := self._recv_command(blocking=False)) is not None:
                                self._logger.debug(f"Detected {command}.")
                                previous_role = self._active_ctx.role

                                if command.type is CommandType.ASSIGN:
                                    # busy, so there is a stream to finish first
                                    self._send_resp_conn.send(False)

                                elif command.type is CommandType.STOP:
                                    self._accept(command, busy=True)
                                    self._logger.info("Received stop command, stopping.")
                                    raise StopIteration()

                                else:
                                    # stop the producer from generating further samples
                                    # and exhaust the current samples generated by the producer
                                    stoppable_stream.stop()
                                    num_samples += sum(
                                        data.num_rows if iter_arrow else 1 for _, data in it
                                    )

                                    # report progress before the role changes, since the
                                    # switch message names the role being left
                                    _report_progress(clock(), num_samples, last_report)
                                    self._send_msg(
                                        MessageType.CTX_SWITCH,
                                        payload=(previous_role.value, command.role.value),
                                    )
                                    self._accept(command, busy=True)
                                    # recreate the work iterable
                                    break

                            now = clock()
                            # send continuous updates to tracker
                            if (num_samples > 0) and (
                                now - last_report > self._progress_report_interval
                            ):
                                # report progress report and reset tracking values
                                _report_progress(now, num_samples, last_report)
                                num_samples, last_report = 0, now

                        else:
                            # producer exhausted
                            stream_exhausted = True
                            self._logger.info("Finished processing current context.")
                            # send final progress update and completion message
                            _report_progress(clock(), num_samples, last_report)
                            self._send_msg(MessageType.CTX_COMPLETE)

                    except StopIteration:
                        # catch stop execution error
                        self._send_msg(MessageType.CTX_CANCELED)
                        stream_exhausted = True
                        done = True

                    except KeyboardInterrupt:
                        self._send_msg(MessageType.CTX_CANCELED)
                        raise

                    except Exception as e:
                        # gracefully handle exceptions without stopping the worker
                        self._logger.error(
                            f"Unexpected error during processing: {str(e)}.", exc_info=True
                        )
                        self._send_msg(
                            MessageType.EXCEPTION,
                            payload={
                                "error_type": str(type(e).__name__),
                                "error_message": str(e),
                                "stack_trace": traceback.format_exc(),
                            },
                        )
                        # Give this context up rather than running it again. The workload
                        # raised on this data and would raise on it again, so retrying is
                        # an endless loop that reports the same failure forever.
                        self._send_msg(MessageType.CTX_CANCELED)
                        stream_exhausted = True

        except KeyboardInterrupt:
            self._logger.warning("Worker interrupted by user.")

        except Exception as e:
            # gracefully handle exception
            self._logger.error(f"Unexpected error during processing: {str(e)}.", exc_info=True)
            self._send_msg(
                MessageType.EXCEPTION,
                payload={
                    "error_type": str(type(e).__name__),
                    "error_message": str(e),
                    "stack_trace": traceback.format_exc(),
                },
            )

        finally:
            try:
                # finalize worker
                self._worker_finalize()
                self._logger.info("Worker finalized successfully.")
            except Exception as e:
                # log exception
                self._logger.error(f"Error finalizing worker: {str(e)}.", exc_info=True)
                self._send_msg(
                    MessageType.EXCEPTION,
                    payload={
                        "error_type": str(type(e).__name__),
                        "error_message": str(e),
                        "stack_trace": traceback.format_exc(),
                    },
                )

            # send done message
            self._send_msg(MessageType.DONE)


class WorkerController(object):
    """Controller for managing worker processes and their roles.

    Handles the assignment of workers to different roles (processors,
    consumers, and producers) and manages task execution and worker
    state transitions.
    """

    def __init__(
        self,
        workers: list[Worker],
        prefetch: int,
        num_shards: int,
        queue: None | _CountedQueue = None,
        queue_it: None | QueueExamplesIterable = None,
        stream_closed: Any = None,
    ) -> None:
        """Initializes the WorkerController with the provided workers and serializer.

        Args:
            workers (list[Worker]): A list of workers to be controlled.
            prefetch (int): The number of samples to prefetch for each worker.
            num_shards (int): The total number of shards to process.
            queue (None | _CountedQueue): The shared data queue, created before the
                workers so they inherit it through `fork`. Defaults to a fresh one, which
                is only useful when there are no real workers to share it with.
            queue_it (None | QueueExamplesIterable): The consumers' view of that queue.
            stream_closed (Any): The event marking the queue as finished. Defaults to a
                fresh one.
        """
        self.prefetch = prefetch
        self.workers = workers
        self.queue = queue if queue is not None else _CountedQueue(maxsize=self.num_workers)
        self.queue_it = (
            queue_it
            if queue_it is not None
            else QueueExamplesIterable(
                self.queue, sentinel=None, timeout=_QUEUE_GET_TIMEOUT, num_shards=num_shards
            )
        )
        self.stream_closed = stream_closed if stream_closed is not None else mp.Event()
        self.standalone_ranks = set()
        self.producer_ranks = set()
        self.consumer_ranks = set()
        self.joined_ranks = set()
        self._logger = logging.getLogger(f"{type(self).__module__}.{type(self).__qualname__}")

    @property
    def num_workers(self) -> int:
        """Returns the number of workers being managed.

        Returns:
            int: Number of workers.
        """
        return len(self.workers)

    @property
    def any_producers(self) -> bool:
        """Checks if there are any workers assigned as producers.

        Returns:
            bool: True if any workers are assigned to the producer role, False otherwise.
        """
        return len(self.producer_ranks) > 0

    def start(self) -> None:
        """Starts all the worker processes. Logs the start of each worker."""
        for worker in self.workers:
            worker.start()

        self._logger.info("All workers started.")

    def assign_shard(self, rank: int, shard_id: int) -> None:
        """Put a worker to work on a shard, on its own.

        Args:
            rank (int): The rank of the worker to assign.
            shard_id (int): The shard for it to process.
        """
        self.workers[rank].send_command(
            Command(CommandType.ASSIGN, role=WorkerRole.STANDALONE, shard_id=shard_id),
            blocking=False,
        )
        self.standalone_ranks.add(rank)
        self._logger.info(f"Assigned shard {shard_id} to worker {rank} as standalone.")

    def assign_consumer(self, rank: int) -> None:
        """Put a worker to work on the shared queue.

        Args:
            rank (int): The rank of the worker to assign.
        """
        self.workers[rank].send_command(
            Command(CommandType.ASSIGN, role=WorkerRole.CONSUMER), blocking=False
        )
        self.consumer_ranks.add(rank)
        self._logger.info(f"Assigned worker {rank} as consumer.")

    def try_switch_standalone_to_producer(self) -> int | None:
        """Attempts to switch a processor to a producer role.

        A worker only accepts a switch while it is part-way through a stream, so a refusal
        means that rank was idle and another should be tried.

        Returns:
            int | None: The rank of the worker if the switch is successful, None otherwise.
        """
        command = Command(CommandType.SWITCH, role=WorkerRole.PRODUCER)
        for rank in self.standalone_ranks:
            # A producer keeps the stream it already has, so it must be promoted from a
            # standalone holding a shard - never from a consumer, which would leave it
            # reading the queue it is supposed to be filling.
            if self.workers[rank].send_command(command, blocking=True):
                self.standalone_ranks.remove(rank)
                self.producer_ranks.add(rank)
                self._logger.info(f"Assigned worker {rank} as producer.")
                return rank
            self._logger.info(f"Worker {rank} did not accept producer command.")

    def try_switch_producer_to_standalone(self) -> int | None:
        """Attempts to switch a producer back to a processor role.

        The worker keeps the shard it was producing from and goes back to writing it out
        itself.

        Returns:
            int | None: The rank of the worker if the switch is successful, None otherwise.
        """
        command = Command(CommandType.SWITCH, role=WorkerRole.STANDALONE)
        for rank in self.producer_ranks:
            if self.workers[rank].send_command(command, blocking=True):
                self.producer_ranks.remove(rank)
                self.standalone_ranks.add(rank)
                self._logger.info(f"Assigned worker {rank} as processor.")
                return rank
            self._logger.info(f"Worker {rank} did not accept processor command.")

    def close_stream(self) -> None:
        """Mark the shared queue as finished, releasing every consumer waiting on it.

        `QueueExamplesIterable` leaves a drained queue either on this flag or, failing
        that, once its `get` has come up empty for the whole timeout. Nothing used to set
        anything, so waiting out 30 s was a consumer's only way to finish - and that is
        the ordinary end of every run, since once the shard pool empties every worker
        still alive is made a consumer.

        A flag rather than a sentinel per consumer, because a sentinel is not addressed:
        any consumer takes any of them, so releasing exactly the set of workers that are
        waiting means counting recipients correctly through role changes, and retrying
        posts that a full queue rejected - at a moment when no messages may be arriving to
        prompt a retry. Setting a flag every consumer can see needs none of that.

        Only safe once no further data can appear; see `_close_stream_if_finished`.
        """
        if not self.stream_closed.is_set():
            self.stream_closed.set()
            self._logger.debug("Marked the data queue as closed.")

    def free_worker(self, rank: int) -> None:
        """Removes the worker from any active roles.

        Args:
            rank (int): The rank of the worker to free.
        """
        self.standalone_ranks -= {rank}
        self.producer_ranks -= {rank}
        self.consumer_ranks -= {rank}

    def stop_worker(self, rank: int) -> None:
        """Sends a stop signal to a worker, indicating that it should cease operation.

        Args:
            rank (int): The rank of the worker to stop.
        """
        self.workers[rank].send_command(Command(CommandType.STOP), blocking=False)

    def stop_all(self) -> None:
        """Sends a stopping signal to all alive workers."""
        # send stop singal to all alive workers
        for rank in range(self.num_workers):
            if self.workers[rank].is_alive():
                self.stop_worker(rank)

    def join_worker(self, rank: int) -> None:
        """Waits for a worker to complete execution and join the main thread.

        Ensures the worker is no longer performing any roles.

        Args:
            rank (int): The rank of the worker to join.
        """
        assert rank not in self.standalone_ranks
        assert rank not in self.producer_ranks
        assert rank not in self.consumer_ranks
        self.workers[rank].join()
        self.joined_ranks.add(rank)

    def assert_all_workers_joined(self) -> None:
        """Asserts that all workers have completed execution and joined the main thread."""
        assert len(self.joined_ranks) == len(self.workers)


class ConsumerProducerBalancer(object):
    """A class to balance the number of producer and consumer workers based on queue state.

    This class monitors the state of the queue and adjusts the number of producer workers
    to ensure efficient processing of items. It uses a callback mechanism to decide whether
    to add or remove producers based on the average block times of the workers and the size
    of the queue.
    """

    class Action(Enum):
        """Enumeration for actions to be taken by the balancer."""

        NO_ACTION = 1
        """No action to be taken."""
        ADD_PRODUCER = 2
        """Indicate to add a producer worker."""
        REMOVE_PRODUCER = 3
        """Indicate to remove a producer worker."""

    def __init__(self, controller: WorkerController, monitor: ProgressMonitor) -> None:
        """Initialize the :class:`ConsumerProducerBalancer` with the given controller and monitor.

        Args:
            controller (WorkerController): The controller managing the workers.
            monitor (ProgressMonitor): The monitor tracking the progress of the system.
        """
        self._controller = controller
        self._monitor = monitor

    def callback(self) -> Action:
        """Evaluate the current queue state and determine the appropriate action.

        This method computes the queue size relative to the target size and the average
        block times for both producer and consumer workers. Based on the calculated values,
        it returns an action to take:
            - :code:`ADD_PRODUCER`: If the queue is too empty and consumers are blocked.
            - :code:`REMOVE_PRODUCER`: If the queue is too full and producers are blocked.
            - :code:`NO_ACTION`: If no adjustments are necessary.

        Returns:
            Action: The action to be taken (add or remove a producer, or no action).
        """
        # compute the queue size with respect to the target queue size
        # which is one item per worker (matching the maximum queue size)
        target_size = self._monitor._item_size * self._controller.num_workers
        queue_size = self._monitor.num_buffered_samples / target_size

        # get registered producer and consumer workers
        registered_producer_workers = self._monitor.get_workers_with_role(WorkerRole.PRODUCER)
        registered_consumer_workers = self._monitor.get_workers_with_role(WorkerRole.CONSUMER)

        # only balance producers if there are consumers
        if len(registered_consumer_workers) == 0:
            return ConsumerProducerBalancer.Action.NO_ACTION

        # get the average block times for producer and consumer group
        producer_block_time = self._monitor.elapsed_time_averages(registered_producer_workers)[
            Stages.FINALIZE
        ]
        consumer_block_time = self._monitor.elapsed_time_averages(registered_consumer_workers)[
            Stages.STREAM
        ]

        # of producers or consumers
        if (queue_size < 0.3) and (consumer_block_time >= 1.3 * producer_block_time):
            # get operations take longer than put operations
            # queue get operation blocks because its empty
            return ConsumerProducerBalancer.Action.ADD_PRODUCER

        elif (queue_size > 0.7) and (producer_block_time >= 1.3 * consumer_block_time):
            # put operations take longer than get operations
            # queue put operation blocks because its full
            return ConsumerProducerBalancer.Action.REMOVE_PRODUCER

        return ConsumerProducerBalancer.Action.NO_ACTION


class DynamicMultiprocessingRunner(BaseRunner):
    """Manages and runs a set of worker processes to handle parallel data processing.

    This class coordinates multiple worker processes to process data in parallel,
    dynamically assigning tasks and handling context changes.

    The processing is carried out in two distinct stages:

    - **Stage 1: Single-Shard Single-Worker**
      Each worker is assigned one dataset shard at a time, with minimal communication overhead,
      maximizing throughput by keeping the workers busy with their assigned tasks.

    - **Stage 2: Single-Shard Multiple-Workers**
      After all shards are assigned, the system transitions into this stage, where multiple
      workers process the same shard, dividing the roles into producers (feeding a queue) and
      consumers (processing data from the queue).

    By transitioning to Stage 2, the system ensures efficient and parallel processing of data,
    optimizing performance and resource usage.
    """

    def __init__(
        self,
        num_workers: int,
        prefetch_factor: int,
        worker_init: Callable[[], Any],
        worker_finalize: Callable[[], Any],
        progress_report_interval: float,
        callback: CallbackManager,
        failure_policy: FailurePolicy = FailurePolicy.FAIL_FAST,
    ) -> None:
        """Initialize the multiprocessing runner.

        Args:
            num_workers (int): The number of worker processes to create for parallel data
                processing.
            prefetch_factor (int): The number of items per worker that should be prefetched in
                Stage 2 when using a queue.
            worker_init (Callable[[], Any]): A callable that will be invoked to initialize each
                worker. This function will run before the worker starts processing data.
            worker_finalize (Callable[[], Any]): A callable that will be invoked to finalize each
                worker. This function will run after the worker has finished processing all data.
            progress_report_interval (float): The time interval, in seconds, between sending
                progress updates.
            callback (CallbackManager): A callback manager that will be invoked at various points
                during the data processing lifecycle.
            failure_policy (FailurePolicy): What to do when the workload raises on a shard.
                Defaults to stopping the run and raising.
        """
        self._num_workers = num_workers
        self._prefetch = prefetch_factor

        self._worker_init = worker_init
        self._worker_finalize = worker_finalize

        self._report_interval = progress_report_interval
        self._callback = callback
        self._failure_policy = failure_policy
        self._failures: list[ShardFailure] = []

        self._logger = logging.getLogger(f"{type(self).__module__}.{type(self).__qualname__}")

    def _prepare_dataset(
        self, ds: IterableDataset
    ) -> tuple[IterableDataset, Callable[[_BaseExamplesIterable], _BaseExamplesIterable]]:
        """Prepare the dataset for processing by separating processing steps.

        Args:
            ds (IterableDataset): The dataset to prepare.

        Returns:
            tuple[IterableDataset, Callable[[_BaseExamplesIterable], _BaseExamplesIterable]]:
            A tuple containing the source dataset and a processor function representing the lazy
            operations applied to the dataset.
        """
        # get the examples iterable from the dataset and replace all rebatch operations
        # with fast rebatch, we do this once at the beginning to make sure the transform
        # also contains only fast rebatch operations
        ex_iterable = FastRebatchedArrowExamplesIterable.replace_rebatch(ds._ex_iterable)

        # TODO: rethink which ex_iterable items to include
        #       maybe just all of them, i.e. all those that have the ex_iterable attribute
        if _is_separable(ex_iterable):
            # collect all processing steps to separate off
            transform = ExamplesIterablePipeline([ex_iterable])
            while _is_separable(transform.src_iterable):
                transform.insert(0, transform.src_iterable)

            self._logger.info(
                f"Separated {len(transform)} processing steps from iterable "
                f"dataset: {str(transform)}"
            )

            ex_iterable = transform.src_iterable

        else:
            # no operations found to separate from the dataset
            transform = identity_func

        # create the source dataset from the examples iterable
        # needs to be arrow formatting to apply rebatching in call to
        # prepare_ex_iterable_from_iteration later
        ds = IterableDataset(
            ex_iterable=ex_iterable, formatting=FormattingConfig(format_type="arrow")
        )
        # replace rebatch operations with fast rebatch
        ex_iterable = ds._prepare_ex_iterable_for_iteration(batch_size=self._prefetch)
        ex_iterable = FastRebatchedArrowExamplesIterable.replace_rebatch(ex_iterable)
        return ex_iterable, transform

    def _handle_message_loop(
        self,
        msg_queue: mp.Queue,
        monitor: ProgressMonitor,
        controller: WorkerController,
        balancer: ConsumerProducerBalancer,
    ) -> None:
        """Handles the message loop for communication between worker processes.

        This method listens for messages from worker processes via the given connection.
        It processes various types of messages related to worker states, including
        readiness, completion, role switching, and progress reporting.

        The pipeline itself is not among the arguments: workers were handed it at
        construction, so the loop only ever tells them which role to take.

        Args:
            msg_queue (mp.Queue): The queue used to receive messages from worker processes.
            monitor (ProgressMonitor): An object responsible for tracking the progress and state
                of the workers and the overall processing.
            controller (WorkerController): The controller managing worker assignments and roles.
            balancer (ConsumerProducerBalancer): An object that balances the number of producer
                and consumer workers based on the current state of the system.

        Returns:
            None: This method operates in a loop until all workers are done, updating their
            status and managing context switches as necessary.
        """
        # mark a specific worker as switching
        # used to rate limit the context switches of workers
        switching_worker: None | int = None
        last_switch = clock()

        def _no_more_data() -> bool:
            """Whether anything could still put data on the shared queue.

            Nothing can once there is no shard waiting to be assigned, no worker
            producing, and no standalone worker left for
            `try_switch_standalone_to_producer` to promote into one.
            """
            return not (
                monitor.any_pending_shards
                or controller.any_producers
                or controller.standalone_ranks
            )

        def _close_stream_if_finished() -> None:
            """Mark the queue closed once the data really has run out.

            A consumer blocked on the shared queue cannot distinguish "empty for now"
            from "empty for good" - that is what the flag is for.

            Note that this says nothing about what is already *buffered*. Data that has
            been produced still has to be worked through, and spare workers joining as
            consumers is what drains it in parallel, so the run is not settled here - the
            flag only says that nothing further will arrive.
            """
            if _no_more_data():
                controller.close_stream()

        done = False
        while not done:
            # receive message from worker
            msg = msg_queue.get()
            msg = orjson.loads(msg)
            # unpack message
            rank: int = msg["rank"]
            msg_type = MessageType(msg["type"])
            payload = msg["payload"]

            # handle message
            if msg_type is MessageType.READY:
                monitor._mark_worker_ready(rank)
                self._logger.debug(f"Worker {rank} ready.")

            elif msg_type is MessageType.DONE:
                controller.join_worker(rank)
                monitor._mark_worker_done(rank)
                self._logger.debug(f"Worker {rank} done.")
                # Every worker reports DONE from its `finally`, so counting them is the
                # one condition that cannot end the loop early. `any_worker_alive` looks
                # equivalent but is not: a worker is only marked alive once it reports
                # READY, so a run short enough to finish before a straggler has started
                # would see "nobody alive" and leave that worker unjoined. That used to
                # be unreachable only because every run sat out the consumer timeout.
                done = len(controller.joined_ranks) == controller.num_workers

            elif msg_type is MessageType.CTX_STARTED:
                # update monitor state
                role = WorkerRole(payload)
                monitor._mark_worker_busy(rank, role)
                # log
                self._logger.info(f"Worker {rank} started running role {role.name}.")

                if rank == switching_worker:
                    # reset switching worker
                    switching_worker = None

            elif msg_type is MessageType.CTX_COMPLETE:
                if monitor.get_worker_role(rank) is WorkerRole.PRODUCER:
                    # try to start another producer shard to replace this one
                    controller.try_switch_standalone_to_producer()

                # get the shard that was processed by the worker
                shard_id = monitor.get_worker_shard(rank)
                # update the controller and monitor state
                controller.free_worker(rank)
                monitor._mark_worker_completed(rank)
                monitor._mark_worker_idling(rank)

                if shard_id is not None:
                    # run the callback
                    self._callback.on_shard_completed(monitor, shard_id)

                _close_stream_if_finished()

            elif msg_type is MessageType.CTX_CANCELED:
                if monitor.get_worker_role(rank) is WorkerRole.PRODUCER:
                    # try to start another producer shard to replace this one
                    controller.try_switch_standalone_to_producer()

                # get the shard that was processed by the worker
                shard_id = monitor.get_worker_shard(rank)
                # update the controller and monitor state
                controller.free_worker(rank)
                monitor._mark_worker_canceled(rank)
                monitor._mark_worker_idling(rank)

                if shard_id is not None:
                    # run the callback
                    self._callback.on_shard_canceled(monitor, shard_id)

                _close_stream_if_finished()

            elif msg_type is MessageType.EXCEPTION:
                # The worker has given this shard up and will ask for another; what the
                # run does about it is this policy's business.
                failure = ShardFailure(
                    rank=rank,
                    shard_id=monitor.get_worker_shard(rank),
                    error_type=payload["error_type"],
                    error_message=payload["error_message"],
                    stack_trace=payload["stack_trace"],
                )
                self._failures.append(failure)
                self._logger.error(f"Processing failed: {failure}\n{failure.stack_trace}")
                self._callback.on_exception(monitor, failure)

                if (self._failure_policy is FailurePolicy.FAIL_FAST) and (not monitor.is_stopping):
                    self._logger.error("Failure policy is fail-fast, stopping all workers.")
                    monitor._mark_as_stopping()
                    self._callback.on_stopping(monitor)
                    controller.stop_all()

            elif msg_type is MessageType.CTX_SWITCH:
                # parse payload
                old_role, new_role = payload
                old_role, new_role = WorkerRole(old_role), WorkerRole(new_role)
                # update monitor state
                monitor._mark_worker_idling(rank)
                monitor._mark_worker_busy(rank, new_role)
                # log
                self._logger.info(
                    f"Worker {rank} switched context from {old_role.name} to {new_role.name}"
                )

            elif msg_type is MessageType.CTX_REPORT:
                monitor._report_progress(rank, report=payload)

                now = clock()
                if (switching_worker is None) and (now - last_switch > 5):
                    # call balancer whenever there is a progress report update
                    action = balancer.callback()

                    if action is ConsumerProducerBalancer.Action.ADD_PRODUCER:
                        # try to convert an active processor to a producer
                        switching_worker = controller.try_switch_standalone_to_producer()
                        last_switch = now

                    elif (action is ConsumerProducerBalancer.Action.REMOVE_PRODUCER) and (
                        len(controller.producer_ranks) > 1
                    ):
                        # try to convert an active producer back to a processor
                        switching_worker = controller.try_switch_producer_to_standalone()
                        last_switch = now

            elif msg_type is MessageType.CTX_REQUEST:
                # worker must be idling
                assert rank in monitor.alive_workers
                assert rank in monitor.idle_workers

                if monitor.is_stopping:
                    # send stop singal
                    controller.stop_worker(rank)

                elif monitor.any_pending_shards:
                    # Stage 1
                    shard_id = monitor.pending_shards.pop()
                    # The worker derives the shard itself; only its id has to travel.
                    controller.assign_shard(rank, shard_id)
                    # run callback
                    self._callback.on_shard_in_progress(monitor, shard_id)

                    # mark shard as assigned to worker
                    monitor._mark_shard_in_progress(rank, shard_id)

                elif _no_more_data() and controller.queue.empty():
                    # Nothing can be produced *and* nothing is left buffered, so a
                    # consumer here would find the stream already closed and complete
                    # again immediately. The queue check is the important half: while data
                    # is still buffered the extra consumers are what drain it in parallel,
                    # which is the whole point of stage 2.
                    _close_stream_if_finished()
                    controller.stop_worker(rank)

                else:
                    # Stage 2

                    # check if there is a producer
                    if not controller.any_producers:
                        controller.try_switch_standalone_to_producer()

                    # assign worker as consumer
                    controller.assign_consumer(rank)

                    # A worker joining as a consumer just as the data runs out has to
                    # find the stream marked closed, or it would sit out the full timeout
                    # on a queue that will never fill again.
                    _close_stream_if_finished()

                    # evenutally all workers are consumers
                    if monitor.alive_workers == controller.consumer_ranks:
                        # this is the signal that gracefully stops the workers
                        monitor._mark_as_stopping()
                        self._callback.on_stopping(monitor)
                        self._logger.info("Stopping criteria reached, gracefully stopping workers.")

    def run(
        self,
        ds: IterableDataset,
        finalizer: Callable[[Any], Any],
        batch_size: None | int = None,
        formatting: None | str = None,
    ) -> None:
        """Execute data processing using the worker processes.

        Args:
            ds: (IterableDataset): The dataset to process.
            finalizer (Callable[[Any], Any]): The function to apply to each sample or batch
                of samples in the dataset as the final processing step.
            batch_size (None | int): The size of each batch to process. If :code:`None`,
                process samples individually. Only affects the finalizer. Defaults to None.
            formatting (None | str): The data format in which samples or batches are provided
                to the finalizer function.
        """
        self._logger.info("Starting data processing.")
        self._failures = []

        num_shards = ds.n_shards
        # prepare the dataset
        src_ds, transform = self._prepare_dataset(ds)
        self._logger.info(f"Dataset prepared with {num_shards} shards.")

        # create a worker message queue
        msg_queue = mp.Queue()
        # Everything below is built *before* the workers, so it reaches them through
        # process creation rather than through the command queue. The data queue cannot be
        # pickled through a pipe at all, and the pipeline is the same several-megabyte
        # object for every worker - neither belongs in a per-command payload.
        data_queue = _CountedQueue(maxsize=self._num_workers)
        stream_closed = mp.Event()
        queue_it = QueueExamplesIterable(
            data_queue,
            sentinel=None,
            timeout=_QUEUE_GET_TIMEOUT,
            num_shards=num_shards,
            closed=stream_closed,
        )
        setup = WorkerSetup(
            transform=transform,
            finalizer=finalizer,
            finalizer_batch_size=batch_size,
            finalizer_formatting=(
                None if formatting is None else FormattingConfig(format_type=formatting)
            ),
            data_source=src_ds,
            num_shards=num_shards,
        )
        # Under `fork` the workers inherit the setup through copy-on-write and nothing is
        # serialised. Otherwise it is dill-pickled once here and the same bytes go to all
        # of them - one dump per run, one load per worker, and nothing per command. Dill
        # because a transform is routinely a closure, which the standard pickler that
        # `spawn` uses for the worker object cannot take.
        worker_setup = setup if mp.get_start_method() == "fork" else dill.dumps(setup)
        # create all workers
        workers = [
            Worker(
                rank=rank,
                num_workers=self._num_workers,
                msg_queue=msg_queue,
                progress_report_interval=self._report_interval,
                worker_init=self._worker_init,
                worker_finalize=self._worker_finalize,
                setup=worker_setup,
                queue_stream=queue_it,
                queue_put=data_queue.put,
                prefetch=self._prefetch,
            )
            for rank in range(self._num_workers)
        ]

        # create controller
        controller = WorkerController(
            workers,
            self._prefetch,
            num_shards,
            queue=data_queue,
            queue_it=queue_it,
            stream_closed=stream_closed,
        )
        controller.start()

        # create the progress monitor, note that the serializer dumps a batch of samples
        # into a single queue element with a batch size set to the prefetch factor
        monitor = ProgressMonitor(num_shards, self._num_workers, controller.queue, self._prefetch)

        # create the consumer producer balancer
        balancer = ConsumerProducerBalancer(controller, monitor)

        # bind handle message loop to all arguments
        message_handler = partial(
            self._handle_message_loop,
            msg_queue=msg_queue,
            monitor=monitor,
            controller=controller,
            balancer=balancer,
        )

        try:
            # run start callback and start message handle loop
            self._callback.on_start(monitor, ds)
            message_handler()

        except KeyboardInterrupt:
            self._logger.warning("Processing interrupted by user.")
            # stop all workers and start message handler again
            controller.stop_all()
            message_handler()

        finally:
            # shutdown
            monitor._mark_as_done()
            self._callback.on_done(monitor)
            controller.assert_all_workers_joined()
            msg_queue.close()

        if self._failures:
            # Raised after the workers are joined, so the failure does not leave processes
            # behind. Under SKIP_SHARD the rest of the dataset was still written; the
            # error names the shards that were not.
            raise ShardProcessingError(self._failures)

        self._logger.info("Runner complete.")
