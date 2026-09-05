import time
from itertools import islice
from queue import Queue
from time import sleep

import pyarrow as pa
import pytest
from datasets import Dataset
from datasets.iterable_dataset import RebatchedArrowExamplesIterable, _BaseExamplesIterable

from crane.core.iterables import (
    FastRebatchedArrowExamplesIterable,
    PreQueueExamplesIterable,
    QueueExamplesIterable,
    StoppableExamplesIterable,
    TimedExamplesIterable,
)


@pytest.fixture
def ex_iterable() -> _BaseExamplesIterable:
    ds = Dataset.from_list([{"field": i} for i in range(20)])
    return (
        ds.to_iterable_dataset()
        .with_format(type="arrow")
        ._prepare_ex_iterable_for_iteration(batch_size=1)
    )


class _RestartingArrowIterable(_BaseExamplesIterable):
    """A source that starts from the beginning every time it is iterated.

    The point of the fix is that :class:`StoppableExamplesIterable` must not ask its source
    for a second pass once the first has run out. A source that counts its passes states
    that directly, without depending on whether some other iterable happens to resume or
    restart.
    """

    def __init__(self, tables: list) -> None:
        super().__init__()
        self.tables = tables
        self.passes = 0

    @property
    def iter_arrow(self):
        return self._iter_arrow

    def _iter_arrow(self):
        self.passes += 1
        for i, table in enumerate(self.tables):
            yield str(i), table

    def __iter__(self):
        for key, table in self._iter_arrow():
            yield key, table.to_pylist()[0]

    @property
    def num_shards(self) -> int:
        return 1

    def _init_state_dict(self) -> dict:
        self._state_dict = {}
        return self._state_dict


class TestStoppableExamplesIterable:
    def test_iter(self, ex_iterable) -> None:
        it = StoppableExamplesIterable(ex_iterable)
        # do a couple of iterations
        assert len([x for x in islice(it, 3)]) == 3
        # stop the iterable
        it.stop()
        assert len([x for x in it]) == 0
        # resume the iteration
        it.resume()
        assert len([x for x in it]) == 17

    def test_iter_arrow(self, ex_iterable) -> None:
        it = StoppableExamplesIterable(ex_iterable)
        # do a couple of iterations
        assert len([x for x in islice(it.iter_arrow(), 3)]) == 3
        # stop the iterable
        it.stop()
        assert len([x for x in it.iter_arrow()]) == 0
        # resume the iteration
        it.resume()
        assert len([x for x in it.iter_arrow()]) == 17

    def test_an_exhausted_stream_is_not_read_again(self) -> None:
        # A worker reuses one of these across role changes, and a change can land on the
        # last batch of a shard - after the source has already run out. Restarting there
        # reads the shard twice, and every one of its rows is written twice: once by the
        # role that was left and once by the role that took over.
        source = _RestartingArrowIterable([pa.table({"field": [i]}) for i in range(3)])
        it = StoppableExamplesIterable(source)

        assert len(list(it.iter_arrow())) == 3
        assert source.passes == 1

        it.stop()
        it.resume()

        assert list(it.iter_arrow()) == []
        assert source.passes == 1, "the exhausted shard was read a second time"

    def test_an_exhausted_stream_is_not_read_again_in_python_format(self) -> None:
        source = _RestartingArrowIterable([pa.table({"field": [i]}) for i in range(3)])
        it = StoppableExamplesIterable(source)

        assert len(list(it)) == 3
        it.stop()
        it.resume()

        assert list(it) == []
        assert source.passes == 1, "the exhausted shard was read a second time"

    def test_a_stop_part_way_through_still_resumes(self) -> None:
        # the other half of the contract: stopping before the end must not lose the rest
        source = _RestartingArrowIterable([pa.table({"field": [i]}) for i in range(4)])
        it = StoppableExamplesIterable(source)

        seen = list(islice(it.iter_arrow(), 2))
        it.stop()
        it.resume()
        seen += list(it.iter_arrow())

        assert len(seen) == 4
        assert source.passes == 1


class TestTimedExamplesIterable:
    def test_iter(self, ex_iterable) -> None:
        it = TimedExamplesIterable(ex_iterable, smoothing=0.5)
        assert it.smooth_time() == 0.0
        assert it.total_time() == 0.0

        # Perform some iterations and simulate processing delays
        for _ in range(3):
            _ = next(iter(it))
            sleep(0.01)  # Simulate a processing delay

        assert it.smooth_time() > 0.0
        assert it.total_time() > 0.0

    def test_iter_arrow(self, ex_iterable) -> None:
        it = TimedExamplesIterable(ex_iterable, smoothing=0.5)
        assert it.smooth_time() == 0.0
        assert it.total_time() == 0.0

        # Simulate processing delays in arrow iteration
        for _ in range(3):
            _ = next(it._iter_arrow())
            sleep(0.01)  # Simulate a processing delay

        assert it.smooth_time() > 0.0
        assert it.total_time() > 0.0


from itertools import islice


@pytest.fixture
def queue() -> Queue:
    """Fixture for creating a multiprocessing queue."""
    return Queue()


class TestPreQueueExamplesIterable:
    def test_iter_arrow(self, ex_iterable) -> None:
        """Test that PreQueueExamplesIterable adds metadata to Arrow tables."""
        prequeue_iterable = PreQueueExamplesIterable(ex_iterable, key="_key")
        for key, pa_table in islice(prequeue_iterable._iter_arrow(), 3):
            metadata = pa_table.schema.metadata
            assert metadata is not None
            assert b"_key" in metadata
            assert metadata[b"_key"] == key.encode()


class TestQueueExamplesIterable:
    def test_queue_integration(self, ex_iterable, queue) -> None:
        """Test that QueueExamplesIterable retrieves data correctly."""
        # Prepare the iterable with PreQueueExamplesIterable
        prequeue_iterable = QueueExamplesIterable.prepare_ex_iterable(ex_iterable)

        # Add Arrow tables to the queue
        for key, pa_table in prequeue_iterable._iter_arrow():
            queue.put(pa_table)

        # Add sentinel to the queue
        sentinel = object()
        queue.put(sentinel)

        # Create a QueueExamplesIterable to consume the queue
        queue_iterable = QueueExamplesIterable(queue=queue, sentinel=sentinel)

        # Iterate over the queue_iterable and verify the results
        for key, pa_table in queue_iterable._iter_arrow():
            metadata = pa_table.schema.metadata
            assert metadata is not None
            assert b"_key" not in metadata  # Key should be removed
            assert isinstance(key, str)

    def test_timeout_behavior(self, queue) -> None:
        """Test the timeout behavior of QueueExamplesIterable."""
        sentinel = object()
        queue_iterable = QueueExamplesIterable(queue=queue, sentinel=sentinel, timeout=0.1)

        # Ensure the iterable doesn't yield anything when the queue is empty
        assert len([x for x in queue_iterable._iter_arrow()]) == 0


class TestFastRebatchedArrowExamplesIterable:
    def test_replace_rebatch(self) -> None:
        ex_iterable = TimedExamplesIterable(None)
        ex_iterable = FastRebatchedArrowExamplesIterable.replace_rebatch(ex_iterable)
        assert isinstance(ex_iterable, TimedExamplesIterable)

        ex_iterable = RebatchedArrowExamplesIterable(None, None)
        ex_iterable = FastRebatchedArrowExamplesIterable.replace_rebatch(ex_iterable)
        assert isinstance(ex_iterable, FastRebatchedArrowExamplesIterable)

        ex_iterable = RebatchedArrowExamplesIterable(None, None)
        ex_iterable = TimedExamplesIterable(ex_iterable)
        ex_iterable = FastRebatchedArrowExamplesIterable.replace_rebatch(ex_iterable)
        assert isinstance(ex_iterable.ex_iterable, FastRebatchedArrowExamplesIterable)

    @pytest.mark.parametrize("drop_last_batch", [True, False])
    def test_iter_arrow(self, ex_iterable: _BaseExamplesIterable, drop_last_batch: bool) -> None:
        print(ex_iterable.iter_arrow)

        ex_iterable = FastRebatchedArrowExamplesIterable(
            ex_iterable, batch_size=8, drop_last_batch=drop_last_batch
        )

        pa_tables = [pa_table for _, pa_table in ex_iterable.iter_arrow()]

        assert pa_tables[0].num_rows == 8
        assert pa_tables[1].num_rows == 8

        if drop_last_batch:
            assert len(pa_tables) == 2
        else:
            assert len(pa_tables) == 3
            assert pa_tables[2].num_rows == 4


class TestQueueExamplesIterableShutdown:
    """How a consumer learns the queue is finished."""

    def test_stops_promptly_once_closed(self):
        import threading
        from queue import Queue as SyncQueue

        closed = threading.Event()
        # A 30s timeout that must not be paid: the flag is what ends the iteration.
        iterable = QueueExamplesIterable(
            SyncQueue(), sentinel=None, timeout=30.0, closed=closed, poll_interval=0.01
        )
        closed.set()

        start = time.perf_counter()
        assert list(iterable._iter_arrow()) == []
        assert time.perf_counter() - start < 5

    def test_keeps_waiting_while_the_stream_is_open(self):
        import threading
        from queue import Queue as SyncQueue

        closed = threading.Event()
        queue = SyncQueue()
        iterable = QueueExamplesIterable(
            queue, sentinel=None, timeout=30.0, closed=closed, poll_interval=0.01
        )

        def close_later():
            time.sleep(0.3)
            closed.set()

        thread = threading.Thread(target=close_later)
        thread.start()
        start = time.perf_counter()
        assert list(iterable._iter_arrow()) == []
        elapsed = time.perf_counter() - start
        thread.join()

        # It waited for the flag rather than leaving the moment the queue looked empty.
        assert 0.25 < elapsed < 5
