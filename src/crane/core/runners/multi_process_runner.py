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
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from queue import Empty, Full
from typing import Any, Callable, Iterable, TypeAlias

import dill
import orjson
import pyarrow as pa
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
from .base import BaseRunner, WorkerProcessingStage, WorkerRole

# shorthands and helper type aliases
Stages: TypeAlias = WorkerProcessingStage

# How long to keep trying to hand a new context to a worker before giving up on that
# particular context. The race this absorbs is the queue feeder thread being momentarily
# behind, which resolves in milliseconds, so keep it short: contexts are also sent from
# the balancer's hot path and a long block there would stall coordination.
_CTX_SEND_TIMEOUT: float = 1.0
_CTX_SEND_RETRY_INTERVAL: float = 0.01

# How long to wait for room when posting an end-of-stream sentinel. Once no more data can
# be produced the queue only ever drains, so a short block is enough to ride out a queue
# that happens to be full at that instant - and it is bounded, because the controller
# gives up after the first failure and retries on its next message instead.
_SENTINEL_PUT_TIMEOUT: float = 0.5

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


@dataclass
class _PrePickled:
    """A context field that was serialised ahead of time.

    `dill.dumps` walking a tokenizer's object graph costs ~0.8 s; copying the 5.7 MB of
    bytes it produces costs a fraction of a millisecond. Every worker is handed the same
    transform, so pickling it once and carrying the resulting buffer through each context
    makes the unavoidable per-worker resends cost a memcpy instead of a full traversal.

    The trade-off is that a part pickled on its own no longer shares structure with the
    rest of the context: an object referenced by both a pre-pickled part and, say, the
    data stream would arrive as two copies. Only the transform and finalizer are treated
    this way, and neither shares state with a shard.
    """

    payload: bytes

    def unwrap(self) -> Any:
        """Deserialise the part."""
        return dill.loads(self.payload)


class _ContextPartCache:
    """Serialises context parts once and hands the same bytes to every worker.

    Keyed on object identity, which is only sound while the object is alive - a
    collected object's id can be handed to another - so a reference is kept alongside
    each entry. The parts worth caching are few and long-lived (one transform, one
    finalizer per role), but role switches build a fresh bound method each time, so the
    cache is capped rather than left to grow for the length of a run.
    """

    def __init__(self, max_entries: int = 32) -> None:
        self._entries: dict[int, tuple[Any, bytes]] = {}
        self._max_entries = max_entries

    def dumps(self, obj: Any) -> _PrePickled:
        """Return `obj` pre-serialised, reusing the bytes if it was seen before."""
        key = id(obj)
        entry = self._entries.get(key)
        if entry is None:
            payload = dill.dumps(obj)
            if len(self._entries) < self._max_entries:
                self._entries[key] = (obj, payload)
        else:
            payload = entry[1]
        return _PrePickled(payload)


def _unwrap_parts(ctx: WorkerContext) -> WorkerContext:
    """Restore any pre-pickled parts of a context received by a worker."""
    changes = {
        field: value.unwrap()
        for field, value in (
            ("data_transform", ctx.data_transform),
            ("data_finalizer", ctx.data_finalizer),
        )
        if isinstance(value, _PrePickled)
    }
    return replace(ctx, **changes) if changes else ctx


@dataclass
class WorkerContext:
    """Dataclass representing the context passed to workers.

    This class defines the context used for setting up a worker's role, data stream,
    transformation function, finalization function, and a completion flag.
    """

    role: None | WorkerRole = None
    """The role assigned to the worker (e.g., producer, processor), or None if the role remains
    unchanged.
    """

    data_stream: None | _BaseExamplesIterable = None
    """An iterable providing the stream of data items that the worker will process."""

    data_transform: None | Callable[[_BaseExamplesIterable], _BaseExamplesIterable] = None
    """A function that wraps the data stream and applies workload on each sample, returning an
    iterable of processed data.
    """

    data_finalizer: None | Callable[[Any], Any] = None
    """A function that applies final processing to each transformed data item, or None if no
    finalization is needed.
    """

    data_finalizer_batch_size: None | int = None
    """The size of each batch to process for the :code:`data_finalizer` function.

    If None, the :code:`data_finalizer` will receive individual samples.
    Otherwise, the :code:`data_finalizer` will receive batches of this size.
    """

    data_finalizer_formatting: None | FormattingConfig = None
    """The data format expected by the data finalizer."""

    stop: bool = False
    """A flag indicating whether the worker should stop processing."""

    def apply_ctx(self, other: WorkerContext) -> None:
        """Applies non-None values of a given context to self.

        Updates the current context's attributes with the values from another WorkerContext.
        If any of the values in the other context are None, the original value is retained.

        Args:
            other (WorkerContext): The context from which to update values.
        """
        if other.role is not None:
            self.role = other.role
        if other.data_stream is not None:
            self.data_stream = other.data_stream
        if other.data_transform is not None:
            self.data_transform = other.data_transform
        if other.data_finalizer is not None:
            self.data_finalizer = other.data_finalizer
            self.data_finalizer_batch_size = other.data_finalizer_batch_size
            self.data_finalizer_formatting = other.data_finalizer_formatting
        self.stop = other.stop

    def __str__(self) -> str:
        """Returns a string representation of the WorkerContext, including the worker role."""
        return f"WorkerContext(worker_role={self.role})"


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
        ctx_part_cache: None | _ContextPartCache = None,
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
            ctx_part_cache (None | _ContextPartCache): Cache of pre-serialised context
                parts, shared with the other workers so that a transform every worker
                receives is pickled once rather than once per worker. Defaults to a
                private cache.
        """
        super(Worker, self).__init__(daemon=True)

        self._rank = rank
        self._num_workers = num_workers
        # connections
        self._msg_queue = msg_queue
        self._ctx_queue = mp.Queue(maxsize=1)
        self._recv_ctx_resp_conn, self._send_ctx_resp_conn = mp.Pipe(duplex=False)
        # rate limit for progress updates
        self._progress_report_interval = progress_report_interval
        # worker initializer and finalizer
        self._worker_init = worker_init
        self._worker_finalize = worker_finalize
        # the worker context including the pipeline
        # to be executed by the worker
        self._ctx = WorkerContext()
        # What this worker was last known to have *received*. Contexts are sent once per
        # shard assignment but these rarely change, and re-serialising them every time is
        # expensive; see `_strip_parts_already_held`.
        self._last_transform: None | Callable = None
        self._last_finalizer: None | tuple = None
        # Shared with the other workers so one transform is pickled once for all of them,
        # not once each. A worker given no cache keeps its own, which is still correct.
        self._ctx_part_cache = ctx_part_cache if ctx_part_cache is not None else _ContextPartCache()

        # create logger
        self._logger = logging.getLogger(f"{type(self).__module__}.{type(self).__qualname__}")

        # Log initialization details
        self._logger.debug(
            f"Created worker with rank {self._rank} of {self._num_workers} total workers."
        )

    def close(self) -> None:
        """Close the Process object.

        This method releases resources held by the Process object.  It is
        an error to call this method if the child process is still running.
        """
        # close connections and queues
        self._recv_ctx_resp_conn.close()
        self._send_ctx_resp_conn.close()
        self._ctx_queue.close()
        # close worker process
        super(Worker, self).close()

    def send_ctx(self, ctx: WorkerContext, blocking: bool = True) -> bool:
        """Send new processing context to the worker.

        Args:
            ctx (WorkerContext): The context to send to the worker.
            blocking (bool): Whether to block and wait for acceptance from the worker.

        Returns:
            bool: Boolean indicating whether the worker accepted the new context.
            If not blocking, it always returns True.
        """
        # clear connection buffer
        while self._recv_ctx_resp_conn.poll():
            self._recv_ctx_resp_conn.recv()

        # serialize and send context
        ctx, transform, finalizer = self._strip_parts_already_held(ctx)
        ctx_bytes = dill.dumps(self._prepickle_parts(ctx))
        if self._send_ctx_bytes(ctx_bytes):
            # Only now is the worker known to hold these. Recording delivery for a
            # context that was dropped would strip them from every later context too,
            # leaving the worker with no pipeline at all.
            self._last_transform = transform
            self._last_finalizer = finalizer

        if blocking:
            # wait for feedback from worker
            accepted = self._recv_ctx_resp_conn.recv()
            # log
            accept_str = "accpeted" if accepted else "refused"
            self._logger.debug(f"Sent new context to worker {self._rank}, worker {accept_str}.")
            # return
            return accepted

        else:
            self._logger.debug(f"Sent new context to worker {self._rank} in non-blocking mode.")
            return True

    def _strip_parts_already_held(
        self, ctx: WorkerContext
    ) -> tuple[WorkerContext, None | Callable, None | tuple]:
        """Drop the parts of a context this worker already holds.

        The transform closes over whatever the workload needs, and for a tokenising
        pipeline that is the tokenizer - measured at 5.7 MB and 0.8 s to pickle. A
        context is sent once per shard assignment, so a 64-shard run spent ~50 s doing
        nothing but re-serialising objects that never changed. Worse, `dill.dumps` runs
        on the controller's single-threaded message loop, so while it worked every other
        worker asking for its next shard was kept waiting: workers were measured idle
        52% of the time. The finalizer is re-sent just as often and costs about 2 MB.

        `WorkerContext.apply_ctx` keeps the previous value for any field arriving as
        None, and the worker's `_ctx` persists across contexts, so omitting an unchanged
        part is exactly equivalent to sending it again.

        Comparison is by identity rather than an unconditional blanking, because a
        producer context carries a *different* transform and finalizer, and a role
        switch must still deliver them. The finalizer additionally carries its batch
        size and formatting, which `apply_ctx` only applies alongside a non-None
        finalizer - so it is only dropped when all three match, and a change to any of
        them re-sends the set.

        Returns:
            tuple[WorkerContext, None | Callable, None | tuple]: the context to send,
            and the transform and finalizer set the worker will hold once it arrives.
        """
        transform = self._last_transform
        finalizer = self._last_finalizer
        dropped: dict[str, None] = {}

        if ctx.data_transform is not None:
            if ctx.data_transform is self._last_transform:
                dropped["data_transform"] = None
            else:
                transform = ctx.data_transform

        if ctx.data_finalizer is not None:
            candidate = (
                ctx.data_finalizer,
                ctx.data_finalizer_batch_size,
                ctx.data_finalizer_formatting,
            )
            held = self._last_finalizer
            if (
                held is not None
                and candidate[0] is held[0]
                and candidate[1] == held[1]
                and candidate[2] == held[2]
            ):
                dropped["data_finalizer"] = None
            else:
                finalizer = candidate

        return (replace(ctx, **dropped) if dropped else ctx), transform, finalizer

    def _prepickle_parts(self, ctx: WorkerContext) -> WorkerContext:
        """Swap the heavy callables for buffers the cache has already serialised.

        What survives `_strip_parts_already_held` is a part the worker genuinely does
        not have yet - the first context of a run, or a role switch. Every worker needs
        the same transform, so without this the controller pickles it once per worker;
        with it, only the first pays.
        """
        prepickled = {
            field: self._ctx_part_cache.dumps(value)
            for field, value in (
                ("data_transform", ctx.data_transform),
                ("data_finalizer", ctx.data_finalizer),
            )
            if value is not None
        }
        return replace(ctx, **prepickled) if prepickled else ctx

    def _send_ctx_bytes(self, ctx_bytes: bytes, timeout: float = _CTX_SEND_TIMEOUT) -> bool:
        """Replace whatever context is queued for the worker with `ctx_bytes`.

        The queue holds at most one pending context, so a stale one is dropped first.
        Draining and then calling `put_nowait` is not enough on its own: a
        `multiprocessing.Queue` is fed by a background thread, so a successful `get()`
        does not free the slot synchronously and an item already in flight can arrive
        between the two calls. `put_nowait` then raises `Full`, which used to escape into
        the controller's message loop and take it down - leaving the workers waiting for a
        context that never came, so they never joined and the run hung.

        Retry within a deadline instead, and treat exhaustion as a dropped context rather
        than a fatal error: the balancer sends these continuously, so the next one will
        carry the same information.

        Returns:
            bool: Whether the context reached the worker's queue.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._ctx_queue.get_nowait()
            except Empty:
                pass

            try:
                self._ctx_queue.put_nowait(ctx_bytes)
                return True
            except Full:
                if time.monotonic() >= deadline:
                    self._logger.warning(
                        f"Could not hand a new context to worker {self._rank} within "
                        f"{timeout}s; the worker is not consuming its context queue. "
                        f"Dropping this context."
                    )
                    return False
                time.sleep(_CTX_SEND_RETRY_INTERVAL)

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

    def _request_new_ctx(self) -> bool:
        """Request new processing context from the main process.

        Returns:
            bool: Whether the worker has been instructed to stop.
        """
        # request new context from main process
        self._logger.debug("Requesting new context from main process.")
        self._send_msg(MessageType.CTX_REQUEST)

        # wait for new context to be received
        while (ctx := self._recv_ctx(blocking=True, timeout=1.0)) is None:
            self._logger.debug("Waiting for worker context...")

        # requesting a new context must provide a new data stream
        if (ctx.data_stream is None) and (not ctx.stop):
            self._send_ctx_resp_conn.send(False)
            return self._request_new_ctx()

        return self._apply_ctx(ctx)

    def _recv_ctx(self, blocking: bool, timeout: float = 1.0) -> WorkerContext | None:
        """Receive a serialized worker context from the internal context queue.

        This method attempts to retrieve a pickled (serialized) worker context object
        from the internal context queue (:code:`_ctx_queue`). The context is deserialized
        using :code:`dill` and returned. If no context is available within the specified
        timeout, the method returns `None`.

        If :code:`keep_producer` is :code:`True`, the current producer is
        preserved, and only the processing stages are updated.

        Args:
            blocking (bool): Whether to block while waiting for a context.
            timeout (float, optional): Maximum time to wait for a context if blocking.
                Defaults to 1.0 seconds.

        Returns:
            WorkerContext | None: The deserialized worker context if available,
            otherwise `None`.
        """
        try:
            ctx_bytes = self._ctx_queue.get(block=blocking, timeout=timeout)
            ctx = _unwrap_parts(dill.loads(ctx_bytes))
            self._logger.debug(f"Received {ctx}.")
            return ctx
        except Empty:
            pass

        self._logger.debug("No worker context received.")  # TODO
        return None

    def _apply_ctx(self, ctx: WorkerContext) -> bool:
        """Apply context to the worker.

        This method updates the worker's internal attributes based on the provided context.
        Each context element (role, producer, processor, finalizer) is conditionally applied,
        meaning if the element is :code:`None`, the corresponding worker attribute remains
        unchanged.

        The method also triggers an event to signal that the context has been successfully received
        and applied. Additionally, it logs the applied role for debugging purposes.

        Returns:
            bool: The :code:`done` flag, which indicates if the process using this context should
            be completed.
        """
        # apply the context to the worker context
        self._ctx.apply_ctx(ctx)
        self._send_ctx_resp_conn.send(True)
        # log
        self._logger.debug(f"Applied {self._ctx}")

        return self._ctx.stop

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
            # initialize worker
            self._worker_init()
            self._send_msg(MessageType.READY)
            self._logger.info("Initialization complete.")

            done = False
            # request a processing context
            while (not done) and (not self._request_new_ctx()):
                stream_exhausted = False
                # create the stoppable data stream iterator here to avoid
                # resetting it when a new context is received during processing
                data_stream = TimedExamplesIterable(self._ctx.data_stream, smoothing=0.1)
                stoppable_stream = StoppableExamplesIterable(data_stream)

                # exhaust producer
                while not stream_exhausted:
                    self._send_msg(MessageType.CTX_STARTED, payload=self._ctx.role.value)
                    self._logger.debug(f"Starting processing of {self._ctx}.")

                    num_samples = 0
                    last_report = clock()

                    stoppable_stream.resume()

                    try:
                        # apply the transformation function to the data stream
                        transformed_stream = self._ctx.data_transform(stoppable_stream)
                        transformed_stream = TimedExamplesIterable(
                            transformed_stream, smoothing=0.1
                        )
                        # apply the finalizer to the transformed stream
                        work_iterator = (
                            IterableDataset(
                                ex_iterable=transformed_stream,
                                formatting=self._ctx.data_finalizer_formatting,
                            )
                            .map(
                                lambda data: (self._ctx.data_finalizer(data) or data),
                                batched=self._ctx.data_finalizer_batch_size is not None,
                                batch_size=self._ctx.data_finalizer_batch_size,
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
                        iter_arrow = (self._ctx.data_finalizer_formatting is not None) and (
                            self._ctx.data_finalizer_formatting.format_type == "arrow"
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

                            # check if the worker was asked to apply a new context
                            if (ctx := self._recv_ctx(blocking=False)) is not None:
                                self._logger.debug("Detected context update request.")

                                if ctx.data_stream is not None:
                                    # new context not accepted
                                    self._send_ctx_resp_conn.send(False)

                                elif ctx.stop:
                                    # stop worker
                                    self._apply_ctx(ctx)
                                    self._logger.info(
                                        "Received stop signal from context, stopping."
                                    )
                                    raise StopIteration()

                                else:
                                    # stop the producer from generating further samples
                                    # and exhaust the current samples generated by the producer
                                    stoppable_stream.stop()
                                    num_samples += sum(
                                        data.num_rows if iter_arrow else 1 for _, data in it
                                    )

                                    # send progress report before switching the context
                                    _report_progress(clock(), num_samples, last_report)
                                    self._send_msg(
                                        MessageType.CTX_SWITCH,
                                        payload=(self._ctx.role.value, ctx.role.value),
                                    )
                                    self._apply_ctx(ctx)
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

    def __init__(self, workers: list[Worker], prefetch: int, num_shards: int) -> None:
        """Initializes the WorkerController with the provided workers and serializer.

        Args:
            workers (list[Worker]): A list of workers to be controlled.
            prefetch (int): The number of samples to prefetch for each worker.
            num_shards (int): The total number of shards to process.
        """
        self.prefetch = prefetch
        self.workers = workers
        self.queue = mp.Manager().Queue(maxsize=self.num_workers)
        self.queue_it = QueueExamplesIterable(
            self.queue, sentinel=None, timeout=30.0, num_shards=num_shards
        )
        self.standalone_ranks = set()
        self.producer_ranks = set()
        self.consumer_ranks = set()
        self.joined_ranks = set()
        # Consumers already handed an end-of-stream sentinel. Sentinels are not addressed
        # to a rank - any consumer takes any of them - so this is really a count of how
        # many are still owed, kept per rank so that `free_worker` can clear it when a
        # worker leaves the role and may later re-enter it.
        self.sentinel_sent_ranks = set()
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

    def create_standalone_worker(
        self,
        rank: int,
        shard: Iterable[pa.Array],
        transform: Callable[[Iterable], Iterable] | None,
        finalizer: Callable[[Any], Any] | None,
        finalizer_batch_size: int | None,
        finalizer_formatting: FormattingConfig | None,
    ) -> None:
        """Assigns a worker the role of processor and provides the processing context.

        Args:
            rank (int): The rank of the worker to assign.
            shard (Iterable): The data shard to be processed.
            transform (Callable[[Iterable], Iterable] | None): The transform function.
            finalizer (Callable[[Any], Any] | None): The finalizer function.
            finalizer_batch_size (int | None): The finalizer batch size.
            finalizer_formatting (FormattingConfig | None): The finalizer formatting config.
        """
        ctx = WorkerContext(
            role=WorkerRole.STANDALONE,
            data_stream=shard,
            data_transform=transform,
            data_finalizer=finalizer,
            data_finalizer_batch_size=finalizer_batch_size,
            data_finalizer_formatting=finalizer_formatting,
        )
        self.workers[rank].send_ctx(ctx, blocking=False)
        self.standalone_ranks.add(rank)
        self._logger.info(f"Assigned worker {rank} as standalone worker.")

    def create_consumer_worker(
        self,
        rank: int,
        transform: Callable[[Iterable], Iterable] | None,
        finalizer: Callable[[Any], Any] | None,
        finalizer_batch_size: int | None,
        finalizer_formatting: FormattingConfig | None,
    ) -> None:
        """Assigns a worker the role of consumer and provides the consumer context.

        Args:
            rank (int): The rank of the worker to assign.
            transform (Callable[[Iterable], Iterable] | None): The tranform function.
            finalizer (Callable[[Any], Any] | None): The finalizer function.
            finalizer_batch_size (int | None): The finalizer batch size.
            finalizer_formatting (FormattingConfig | None): The finalizer formatting config.
        """
        ctx = WorkerContext(
            role=WorkerRole.CONSUMER,
            data_stream=self.queue_it,
            data_transform=transform,
            data_finalizer=finalizer,
            data_finalizer_batch_size=finalizer_batch_size,
            data_finalizer_formatting=finalizer_formatting,
        )
        self.workers[rank].send_ctx(ctx, blocking=False)
        self.consumer_ranks.add(rank)
        self._logger.info(f"Assigned worker {rank} as consumer.")

    def try_switch_standalone_to_producer(self) -> int | None:
        """Attempts to switch a processor to a producer role.

        If successful, the processor rank is removed from the processor set and added to the
        producer set.

        Returns:
            int | None: The rank of the worker if the switch is successful, None otherwise.
        """
        ctx_update = WorkerContext(
            role=WorkerRole.PRODUCER,
            data_transform=QueueExamplesIterable.prepare_ex_iterable,
            data_finalizer=self.queue.put,
            data_finalizer_batch_size=self.prefetch,
            data_finalizer_formatting=FormattingConfig(format_type="arrow"),
        )
        for rank in self.standalone_ranks:
            if self.workers[rank].send_ctx(ctx_update, blocking=True):
                self.standalone_ranks.remove(rank)
                self.producer_ranks.add(rank)
                self._logger.info(f"Assigned worker {rank} as producer.")
                return rank
            self._logger.info(f"Worker {rank} did not accept producer context.")

    def try_switch_producer_to_standalone(
        self,
        transform: Callable[[Iterable], Iterable] | None,
        finalizer: Callable[[Any], Any] | None,
        finalizer_batch_size: int | None,
        finalizer_formatting: FormattingConfig | None,
    ) -> int | None:
        """Attempts to switch a producer to a processor role.

        If successful, the producer rank is removed from the producer set and added to the
        processor set.

        Args:
            transform (Callable[[Iterable], Iterable] | None): The transform function.
            finalizer (Callable[[Any], Any] | None): The finalizer function.
            finalizer_batch_size (int | None): The finalizer batch size.
            finalizer_formatting (FormattingConfig | None): The finalizer formatting config.

        Returns:
            int | None: The rank of the worker if the switch is successful, None otherwise.
        """
        ctx_update = WorkerContext(
            role=WorkerRole.STANDALONE,
            data_transform=transform,
            data_finalizer=finalizer,
            data_finalizer_batch_size=finalizer_batch_size,
            data_finalizer_formatting=finalizer_formatting,
        )
        for rank in self.producer_ranks:
            if self.workers[rank].send_ctx(ctx_update, blocking=True):
                self.producer_ranks.remove(rank)
                self.standalone_ranks.add(rank)
                self._logger.info(f"Assigned worker {rank} as processor.")
                return rank
            self._logger.info(f"Worker {rank} did not accept processor context.")

    def signal_no_more_data(self) -> int:
        """Post the queue's end-of-stream sentinel, one for each consumer.

        `QueueExamplesIterable` leaves a drained queue either on the sentinel or, failing
        that, once its `get` times out after 30 s - and nothing anywhere ever posted the
        sentinel, so waiting out the timeout in full was a consumer's only way to finish.
        That is the ordinary end of every run rather than an edge case: once the shard
        pool empties, every worker still alive is made a consumer, so a run paid 30 s per
        consumer round *after* its real work was already done. Measured on a 4-worker
        run whose 8 shards were finished at t=18 s, the run did not end until t=76 s.

        Only safe once no further data can appear - the caller establishes that; see
        `_release_finished_consumers`.

        A sentinel that cannot be posted must not simply be forgotten. The queue holds one
        item per worker, so at the moment the end is detected it may still be full of real
        data; dropping the attempt then leaves those consumers to discover the end by
        timing out, which is exactly the 30 s stall this exists to prevent. It showed up in
        benchmarks as a rare but severe collapse - most runs at 5185 documents/s, an
        occasional one at 319, the difference being a multiple of 30 s. So each consumer is
        tracked until it has actually been given one, a short block absorbs a queue that is
        merely busy, and whatever is still owed is retried on the controller's next message.

        Returns:
            int: The number of sentinels posted by this call.
        """
        posted = 0
        for rank in tuple(self.consumer_ranks):
            if rank in self.sentinel_sent_ranks:
                continue
            try:
                self.queue.put(self.queue_it.sentinel, timeout=_SENTINEL_PUT_TIMEOUT)
            except Full:
                # Still busy. Leave the rest owed rather than blocking the controller
                # any longer; the next message retries.
                self._logger.debug("Data queue full while posting sentinels; will retry.")
                break
            self.sentinel_sent_ranks.add(rank)
            posted += 1

        if posted > 0:
            self._logger.debug(f"Posted {posted} end-of-stream sentinels to the data queue.")

        return posted

    @property
    def consumers_awaiting_sentinel(self) -> set[int]:
        """Consumers that are owed an end-of-stream sentinel and have not had one."""
        return self.consumer_ranks - self.sentinel_sent_ranks

    def free_worker(self, rank: int) -> None:
        """Removes the worker from any active roles.

        Args:
            rank (int): The rank of the worker to free.
        """
        self.standalone_ranks -= {rank}
        self.producer_ranks -= {rank}
        self.consumer_ranks -= {rank}
        # If it becomes a consumer again it needs a sentinel of its own.
        self.sentinel_sent_ranks -= {rank}

    def stop_worker(self, rank: int) -> None:
        """Sends a stop signal to a worker, indicating that it should cease operation.

        Args:
            rank (int): The rank of the worker to stop.
        """
        self.workers[rank].send_ctx(WorkerContext(stop=True), blocking=False)

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
        """
        self._num_workers = num_workers
        self._prefetch = prefetch_factor

        self._worker_init = worker_init
        self._worker_finalize = worker_finalize

        self._report_interval = progress_report_interval
        self._callback = callback

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
        src_ds: IterableDataset,
        transform: Callable[[_BaseExamplesIterable], _BaseExamplesIterable],
        finalizer: Callable[[Any], Any],
        finalizer_batch_size: None | int,
        finalizer_formatting: None | FormattingConfig,
        msg_queue: mp.Queue,
        monitor: ProgressMonitor,
        controller: WorkerController,
        balancer: ConsumerProducerBalancer,
    ) -> None:
        """Handles the message loop for communication between worker processes.

        This method listens for messages from worker processes via the given connection.
        It processes various types of messages related to worker states, including
        readiness, completion, context switching, and progress reporting.

        Args:
            src_ds (IterableDataset): The source dataset from which shards are drawn for
                processing.
            transform (Callable[[Iterable[T]], Iterable[U]]): A transformation function that
                processes a shard of data.
            finalizer (Callable[[Any], Any]): A function that finalizes each sample or batch
                after it has been transformed.
            finalizer_batch_size (int | None): The finalizer batch size.
            finalizer_formatting (FormattingConfig | None): The finalizer formatting config.
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

        def _release_finished_consumers() -> None:
            """Tell the consumers the data has run out, when it actually has.

            A consumer blocked on the shared queue cannot distinguish "empty for now"
            from "empty for good" - that is what the sentinel is for. The condition for
            posting it is that nothing can produce again: no shard is waiting to be
            assigned, no worker is producing, and no standalone worker is left that
            `try_switch_standalone_to_producer` could still promote into one.
            """
            if monitor.any_pending_shards:
                return
            if controller.any_producers or controller.standalone_ranks:
                return
            controller.signal_no_more_data()

        def _retry_owed_sentinels() -> None:
            """Deliver any sentinel a full queue prevented from going out earlier.

            Cheap to call on every message, and it has to be: once every worker is a
            consumer blocked on a drained queue, no further message will arrive to
            prompt a retry, so the retries have to happen while messages are still
            flowing.
            """
            if controller.consumers_awaiting_sentinel:
                _release_finished_consumers()

        done = False
        while not done:
            # receive message from worker
            msg = msg_queue.get()
            msg = orjson.loads(msg)
            # unpack message
            rank: int = msg["rank"]
            msg_type = MessageType(msg["type"])
            payload = msg["payload"]

            _retry_owed_sentinels()

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

                _release_finished_consumers()

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

                _release_finished_consumers()

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
                        switching_worker = controller.try_switch_producer_to_standalone(
                            transform=transform,
                            finalizer=finalizer,
                            finalizer_batch_size=finalizer_batch_size,
                            finalizer_formatting=finalizer_formatting,
                        )
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
                    # get shard and send processor context to worker
                    shard = src_ds.shard_data_sources(monitor.num_shards, shard_id)
                    controller.create_standalone_worker(
                        rank=rank,
                        shard=shard,
                        transform=transform,
                        finalizer=finalizer,
                        finalizer_batch_size=finalizer_batch_size,
                        finalizer_formatting=finalizer_formatting,
                    )
                    # run callback
                    self._callback.on_shard_in_progress(monitor, shard_id)

                    # mark shard as assigned to worker
                    monitor._mark_shard_in_progress(rank, shard_id)
                    self._logger.info(f"Assigned shard {shard_id} to worker {rank}.")

                else:
                    # Stage 2

                    # check if there is a producer
                    if not controller.any_producers:
                        controller.try_switch_standalone_to_producer()

                    # assign worker as consumer
                    controller.create_consumer_worker(
                        rank=rank,
                        transform=transform,
                        finalizer=finalizer,
                        finalizer_batch_size=finalizer_batch_size,
                        finalizer_formatting=finalizer_formatting,
                    )

                    # A worker joining as a consumer after the data has already run
                    # out needs its own sentinel, or it would sit out the full timeout
                    # on a queue that will never fill again.
                    _release_finished_consumers()

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

        num_shards = ds.n_shards
        # prepare the dataset
        src_ds, transform = self._prepare_dataset(ds)
        self._logger.info(f"Dataset prepared with {num_shards} shards.")

        # create a worker message queue
        msg_queue = mp.Queue()
        # Shared across the workers on purpose: every one of them is handed the same
        # transform, and pickling it once instead of once per worker is most of what
        # keeps the controller responsive at high worker counts.
        ctx_part_cache = _ContextPartCache()
        # create all workers
        workers = [
            Worker(
                rank=rank,
                num_workers=self._num_workers,
                msg_queue=msg_queue,
                progress_report_interval=self._report_interval,
                worker_init=self._worker_init,
                worker_finalize=self._worker_finalize,
                ctx_part_cache=ctx_part_cache,
            )
            for rank in range(self._num_workers)
        ]

        # create controller
        controller = WorkerController(workers, self._prefetch, num_shards)
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
            src_ds=src_ds,
            transform=transform,
            finalizer=finalizer,
            finalizer_batch_size=batch_size,
            finalizer_formatting=(
                None if formatting is None else FormattingConfig(format_type=formatting)
            ),
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

        self._logger.info("Runner complete.")
