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
