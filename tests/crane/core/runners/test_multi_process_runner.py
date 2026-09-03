import json
import multiprocessing as mp
import time
from queue import Full
from unittest.mock import MagicMock, call, patch

import datasets
import dill
import pytest
from datasets import Dataset
from datasets.iterable_dataset import SelectColumnsIterable

from crane.core.callbacks.base import CallbackManager
from crane.core.runners.base import WorkerProcessingStage, WorkerRole
from crane.core.runners.multi_process_runner import (
    ConsumerProducerBalancer,
    DynamicMultiprocessingRunner,
    MessageType,
    Worker,
    WorkerContext,
    WorkerController,
    _ContextPartCache,
    _PrePickled,
    _unwrap_parts,
)
from tests.third_party.sharedmock.mock import SharedMock


class TestWorker:
    @pytest.fixture
    def worker_init(self):
        return MagicMock()

    @pytest.fixture
    def worker_finalizer(self):
        return MagicMock()

    @pytest.fixture(scope="session")
    def msg_queue(self):
        q = mp.Queue()
        yield q
        q.close()

    @pytest.fixture
    def transform(self):
        return MagicMock(side_effect=lambda x: x)

    @pytest.fixture
    def data_stream(self):
        dummy_data = [
            {"a": 0, "b": [1, 2, 3, 4]},
            {"a": 1, "b": [5, 6]},
            {"a": 1, "b": [7, 8, 9, 10]},
        ]
        ds = Dataset.from_list(dummy_data)
        ds = ds.to_iterable_dataset(3)
        return ds._ex_iterable

    @pytest.fixture
    def context(self, data_stream, transform):
        return WorkerContext(
            role=WorkerRole.STANDALONE,
            data_stream=data_stream,
            data_transform=transform,
            data_finalizer=MagicMock(return_value=None),
            stop=False,
        )

    @pytest.fixture
    def worker(self, msg_queue, worker_init, worker_finalizer, context):
        worker = Worker(
            rank=0,
            num_workers=1,
            msg_queue=msg_queue,
            progress_report_interval=0.0,
            worker_init=worker_init,
            worker_finalize=worker_finalizer,
        )
        worker._ctx = context
        return worker

    def test_request_new_ctx(self, msg_queue, worker):
        ctx = WorkerContext(
            role=WorkerRole.STANDALONE,
            data_stream="STREAM",
            data_transform="TRANSFORM",
            data_finalizer="FINALIZE",
            stop=False,
        )
        # mock context conn receiver to avoid deadlock
        worker._recv_ctx_resp_conn.recv = MagicMock()
        # send new context before worker request to avoid deadlock in worker
        worker.send_ctx(ctx)
        worker._recv_ctx_resp_conn.recv.assert_called_once()

        # request new context
        worker._request_new_ctx()

        # make sure the worker send a request context message
        msg_bytes = msg_queue.get(timeout=1.0)
        msg = json.loads(msg_bytes.decode("utf-8"))
        assert msg["rank"] == worker._rank
        assert msg["type"] == MessageType.CTX_REQUEST.value

        # check if new context was applied
        assert worker._ctx == ctx

    def test_send_ctx_survives_full_queue(self, worker):
        # A multiprocessing queue is fed by a background thread, so an item can land in
        # the slot between the drain and the put. That used to raise `Full` out of
        # `send_ctx` and kill the controller's message loop, leaving workers waiting for
        # a context that never arrived - the run then hung instead of finishing.
        ctx = WorkerContext(
            role=WorkerRole.STANDALONE,
            data_stream="STREAM",
            data_transform="TRANSFORM",
            data_finalizer="FINALIZE",
            stop=False,
        )
        worker._recv_ctx_resp_conn.recv = MagicMock()

        real_put_nowait = worker._ctx_queue.put_nowait
        calls = []

        def flaky_put_nowait(item):
            calls.append(item)
            if len(calls) == 1:
                raise Full()
            real_put_nowait(item)

        worker._ctx_queue.put_nowait = flaky_put_nowait
        worker.send_ctx(ctx)

        assert len(calls) == 2, "should have retried after the queue reported itself full"
        # Heavy parts travel pre-serialised; `_recv_ctx` restores them, so the context
        # the worker ends up with is what has to match.
        assert _unwrap_parts(dill.loads(worker._ctx_queue.get(timeout=1.0))) == ctx

    def test_send_ctx_gives_up_without_raising(self, worker):
        # A worker that never drains its queue must not take the controller down with it.
        ctx = WorkerContext(
            role=WorkerRole.STANDALONE,
            data_stream="STREAM",
            data_transform="TRANSFORM",
            data_finalizer="FINALIZE",
            stop=False,
        )
        worker._recv_ctx_resp_conn.recv = MagicMock()
        worker._ctx_queue.put_nowait = MagicMock(side_effect=Full())

        worker.send_ctx(ctx)  # must return rather than raise

        assert worker._ctx_queue.put_nowait.call_count > 1

    @patch("crane.core.runners.multi_process_runner.set_worker_info")
    def test_run(self, mock_set_worker_info, worker, data_stream, transform):
        # mock request new context
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        # run worker
        worker.run()

        mock_set_worker_info.assert_called_once()
        # make sure all samples have been processed
        transform.assert_called_once()
        worker._ctx.data_finalizer.assert_has_calls(
            [call(x) for _, x in data_stream], any_order=True
        )
        worker._worker_init.assert_called_once()
        worker._worker_finalize.assert_called_once()

    @patch("crane.core.runners.multi_process_runner.set_worker_info")
    def test_run_with_ctx_update(self, mock_set_worker_info, worker, data_stream, transform):
        # mock request new and check context
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        # mock context connection
        worker._recv_ctx_resp_conn = MagicMock()
        worker._recv_ctx_resp_conn.poll = MagicMock(return_value=True)
        # mock receive context function
        ctx_update = WorkerContext(role=WorkerRole.STANDALONE, stop=False)
        worker._recv_ctx = MagicMock(return_value=ctx_update)

        worker.run()

        mock_set_worker_info.assert_called_once()
        # make sure all samples have been processed
        transform.assert_called()
        assert len(worker._recv_ctx.mock_calls) == 3
        worker._ctx.data_finalizer.assert_has_calls(
            [call(x) for _, x in data_stream], any_order=True
        )
        worker._worker_init.assert_called_once()
        worker._worker_finalize.assert_called_once()

    @patch("crane.core.runners.multi_process_runner.set_worker_info")
    def test_run_with_abort(self, mock_set_worker_info, worker, data_stream, transform):
        # mock request new and check context
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        # mock receive context function
        ctx_update = WorkerContext(role=WorkerRole.STANDALONE, stop=True)
        worker._recv_ctx = MagicMock(return_value=ctx_update)

        worker.run()

        transform.assert_called_once()
        mock_set_worker_info.assert_called_once()
        # make sure all samples have been processed
        assert len(worker._recv_ctx.mock_calls) == 1
        worker._ctx.data_finalizer.assert_has_calls(
            [call({"a": 0, "b": [1, 2, 3, 4]})], any_order=True
        )


class TestConsumerProducerBalancer(object):
    @pytest.fixture
    def mock_controller(self):
        """Fixture to create a mocked WorkerController."""
        controller = MagicMock()
        controller.num_workers = 10
        return controller

    @pytest.fixture
    def mock_monitor(self):
        """Fixture to create a mocked ProgressMonitor."""
        monitor = MagicMock()
        monitor.num_buffered_samples = 100  # Default queue size
        monitor._item_size = 10  # Size of queue items
        return monitor

    @pytest.fixture
    def balancer(self, mock_controller, mock_monitor):
        """Fixture to create the ConsumerProducerBalancer with mocked dependencies."""
        return ConsumerProducerBalancer(controller=mock_controller, monitor=mock_monitor)

    def test_callback_add_producer(self, balancer, mock_monitor):
        """Test callback when producers should be added."""
        # Simulate a low queue size (below 30%) and longer consumer block time
        mock_monitor.num_buffered_samples = 20
        mock_monitor.get_workers_with_role.side_effect = {
            WorkerRole.STANDALONE: {1, 2, 3},
            WorkerRole.CONSUMER: {4, 5, 6},
            WorkerRole.PRODUCER: {7, 8, 9},
        }.get
        mock_monitor.elapsed_time_averages.side_effect = lambda _: {
            WorkerProcessingStage.FINALIZE: 1.0,  # Producer block time
            WorkerProcessingStage.STREAM: 2.0,  # Consumer block time
        }

        action = balancer.callback()
        assert action == ConsumerProducerBalancer.Action.ADD_PRODUCER

    def test_callback_remove_producer(self, balancer, mock_monitor):
        """Test callback when producers should be removed."""
        # Simulate a high queue size (above 70%) and longer producer block time
        mock_monitor.num_buffered_samples = 80
        mock_monitor.get_workers_with_role.side_effect = {
            WorkerRole.STANDALONE: {1, 2, 3},
            WorkerRole.CONSUMER: {4, 5, 6},
            WorkerRole.PRODUCER: {7, 8, 9},
        }.get
        mock_monitor.elapsed_time_averages.side_effect = lambda _: {
            WorkerProcessingStage.FINALIZE: 2.0,  # Producer block time
            WorkerProcessingStage.STREAM: 1.0,  # Consumer block time
        }

        action = balancer.callback()
        assert action == ConsumerProducerBalancer.Action.REMOVE_PRODUCER

    def test_callback_no_action(self, balancer, mock_monitor):
        """Test callback when no action should be taken."""
        # Simulate balanced queue size and block times
        mock_monitor.num_buffered_samples = 50
        mock_monitor.get_workers_with_role.side_effect = {
            WorkerRole.STANDALONE: {1, 2, 3},
            WorkerRole.CONSUMER: {4, 5, 6},
            WorkerRole.PRODUCER: {7, 8, 9},
        }.get
        mock_monitor.elapsed_time_averages.side_effect = lambda _: {
            WorkerProcessingStage.FINALIZE: 1.0,  # Producer block time
            WorkerProcessingStage.STREAM: 1.0,  # Consumer block time
        }

        action = balancer.callback()
        assert action == ConsumerProducerBalancer.Action.NO_ACTION

        # Simulate balanced queue size and block times
        mock_monitor.get_workers_with_role.side_effect = {
            WorkerRole.STANDALONE: {1, 2, 3},
            WorkerRole.CONSUMER: set(),  # no consumers -> no action
            WorkerRole.PRODUCER: {7, 8, 9},
        }.get
        mock_monitor.elapsed_time_averages.side_effect = lambda _: {
            WorkerProcessingStage.FINALIZE: 2.0,  # Producer block time
            WorkerProcessingStage.STREAM: 1.0,  # Consumer block time
        }

        action = balancer.callback()
        assert action == ConsumerProducerBalancer.Action.NO_ACTION


def _double_fn(x):
    return {"obj": x["obj"] * 2}


def _plus_one_fn(x):
    return {"obj": x["obj"] + 1}


def _iter_chain(ex_iterable):
    """Yield an examples-iterable and every step beneath it."""
    node = ex_iterable
    while node is not None:
        yield node
        node = getattr(node, "ex_iterable", None)


class TestDynamicMultiprocessingRunner:
    @pytest.fixture
    def ds(self):
        # create mock dataset
        samples = {"obj": [i for i in range(20)]}
        ds = Dataset.from_dict(samples)
        ds = ds.to_iterable_dataset(3)
        return ds

    @pytest.fixture
    def runner(self):
        # run dynamic multiprocessing runner
        return DynamicMultiprocessingRunner(
            num_workers=2,
            prefetch_factor=8,
            worker_init=SharedMock(),
            worker_finalize=SharedMock(),
            progress_report_interval=0.0,
            callback=CallbackManager([]),
        )

    @pytest.mark.parametrize("num_shards", [2, 3])
    def test_run(self, num_shards, runner):
        # create mock dataset
        samples = {"obj": [i for i in range(20)]}
        ds = Dataset.from_dict(samples)
        ds = ds.to_iterable_dataset(num_shards)
        # create mock processor and finalizer
        fn = SharedMock()
        runner.run(ds, fn)
        # make sure all samples have been processed
        fn.assert_has_calls([call(sample) for sample in ds], same_order=False)

    @pytest.mark.parametrize("num_shards", [2, 3])
    @pytest.mark.parametrize("with_features", [False, True])
    def test_run_with_map(self, num_shards, with_features, runner):
        # a lazy `.map` inserts a `FormattedExamplesIterable` into the chain (newer
        # `datasets`); the worker must still be able to drive the separated source
        # through the arrow path when it is wrapped in a StoppableExamplesIterable.
        samples = {"obj": [i for i in range(20)]}
        ds = Dataset.from_dict(samples)
        ds = ds.to_iterable_dataset(num_shards)
        if with_features:
            features = datasets.Features({"obj": datasets.Value("int64")})
            ds = ds.map(_plus_one_fn, features=features)
        else:
            ds = ds.map(_plus_one_fn)

        fn = SharedMock()
        runner.run(ds, fn)
        # make sure the mapped samples have been processed
        fn.assert_has_calls([call(sample) for sample in ds], same_order=False)

    def test_run_with_keyboard_interrupt(self, runner, ds):
        def raise_exc(sample):
            raise KeyboardInterrupt()

        runner.run(ds, raise_exc)

    @pytest.mark.parametrize("num_shards", [2, 3])
    def test_run_with_map_removing_columns(self, num_shards, runner):
        # `.map(remove_columns=...)` puts a `SelectColumnsIterable` between the maps. If
        # the split stops there, every step below stays with the source and the run fails
        # sharding an arrow-formatted map that is still attached to it.
        samples = {"obj": [i for i in range(20)], "drop_me": ["x"] * 20}
        ds = Dataset.from_dict(samples).to_iterable_dataset(num_shards)
        ds = ds.map(_plus_one_fn, remove_columns=["drop_me"]).map(_double_fn)

        fn = SharedMock()
        runner.run(ds, fn)
        fn.assert_has_calls([call(sample) for sample in ds], same_order=False)

    def test_prepare_dataset_separates_projection_above_processing(self, runner):
        # a projection with real work beneath it must be separated, otherwise that work
        # is stranded in the producers
        ds = Dataset.from_dict({"obj": list(range(8)), "drop_me": ["x"] * 8})
        ds = ds.to_iterable_dataset(2).map(_plus_one_fn, remove_columns=["drop_me"])

        src_ex_it, pipeline = runner._prepare_dataset(ds)

        assert not any(
            isinstance(step, SelectColumnsIterable) for step in _iter_chain(src_ex_it)
        ), "projection should have been separated off the source"
        assert [v for _, v in pipeline(src_ex_it)] == list(ds)

    def test_prepare_dataset_keeps_projection_directly_on_source(self, runner):
        # nothing separable beneath it, so keeping it in the producer is what limits how
        # many columns travel through the queue
        ds = Dataset.from_dict({"obj": list(range(8)), "drop_me": ["x"] * 8})
        ds = ds.to_iterable_dataset(2).select_columns(["obj"])

        src_ex_it, pipeline = runner._prepare_dataset(ds)

        assert any(
            isinstance(step, SelectColumnsIterable) for step in _iter_chain(src_ex_it)
        ), "a projection sitting directly on the source should stay with it"
        assert [v for _, v in pipeline(src_ex_it)] == list(ds)

    @pytest.mark.parametrize("nested", [0, 1, 2])
    def test_prepare_dataset(self, nested, runner, ds):
        # apply map function
        mapped_ds = ds
        for _ in range(nested):
            mapped_ds = mapped_ds.map(_double_fn)

        src_ex_it, pipeline = runner._prepare_dataset(mapped_ds)

        # test pipeline output
        expected = [v for _, v in pipeline(src_ex_it)]
        actual = list(mapped_ds)
        assert actual == expected


def _a_transform(ex_iterable):
    return ex_iterable


def _another_transform(ex_iterable):
    return ex_iterable


def _standalone_ctx(transform=None, finalizer=None, batch_size=None, formatting=None):
    return WorkerContext(
        role=WorkerRole.STANDALONE,
        data_stream="STREAM",
        data_transform=transform,
        data_finalizer=finalizer,
        data_finalizer_batch_size=batch_size,
        data_finalizer_formatting=formatting,
    )


def _capture_sent(store):
    """Record what went on the wire, with pre-pickled parts left as they were sent."""

    def capture(ctx_bytes, **kwargs):
        store.append(dill.loads(ctx_bytes))
        return True

    return capture


class TestWorkerContextReuse:
    """A worker is told its transform once, not once per shard."""

    @pytest.fixture
    def worker(self):
        queue = mp.Queue()
        worker = Worker(
            rank=0,
            num_workers=1,
            msg_queue=queue,
            progress_report_interval=0.0,
            worker_init=MagicMock(),
            worker_finalize=MagicMock(),
        )
        yield worker
        queue.close()

    def test_omits_a_transform_the_worker_already_has(self, worker):
        # The transform closes over the workload - a tokenizer, in the case that
        # motivated this - and re-pickling it on every shard assignment made the
        # single-threaded controller the bottleneck. `apply_ctx` keeps the previous
        # value for any field arriving as None, so leaving it out is equivalent.
        sent = []
        with patch.object(worker, "_send_ctx_bytes", side_effect=_capture_sent(sent)):
            worker.send_ctx(_standalone_ctx(_a_transform), blocking=False)
            worker.send_ctx(_standalone_ctx(_a_transform), blocking=False)

        assert sent[0].data_transform is not None
        assert sent[1].data_transform is None

    def test_sends_a_transform_that_actually_changed(self, worker):
        # A producer context carries a different transform, so the check has to be by
        # identity rather than a blanket blanking of every repeat context.
        sent = []
        with patch.object(worker, "_send_ctx_bytes", side_effect=_capture_sent(sent)):
            worker.send_ctx(_standalone_ctx(_a_transform), blocking=False)
            worker.send_ctx(_standalone_ctx(_another_transform), blocking=False)

        assert sent[0].data_transform is not None
        assert sent[1].data_transform is not None

    def test_resends_a_transform_whose_context_was_dropped(self, worker):
        # A dropped context never reached the worker, so it does not hold that transform
        # and the next context must still carry it - otherwise it is left with no
        # pipeline at all.
        with patch.object(worker, "_send_ctx_bytes", return_value=False):
            worker.send_ctx(_standalone_ctx(_a_transform), blocking=False)

        sent = []
        with patch.object(worker, "_send_ctx_bytes", side_effect=_capture_sent(sent)):
            worker.send_ctx(_standalone_ctx(_a_transform), blocking=False)

        assert sent[0].data_transform is not None


class TestConsumerShutdown:
    """Consumers are told the data ran out instead of inferring it from a timeout."""

    def test_signal_no_more_data_posts_one_sentinel_per_consumer(self):
        controller = WorkerController(
            workers=[MagicMock(), MagicMock(), MagicMock()], prefetch=8, num_shards=2
        )
        controller.consumer_ranks = {0, 1}

        assert controller.signal_no_more_data() == 2
        assert controller.queue.get_nowait() is controller.queue_it.sentinel
        assert controller.queue.get_nowait() is controller.queue_it.sentinel
        assert controller.queue.empty()

    def test_signal_no_more_data_is_a_no_op_without_consumers(self):
        controller = WorkerController(workers=[MagicMock()], prefetch=8, num_shards=1)

        assert controller.signal_no_more_data() == 0
        assert controller.queue.empty()

    def test_run_does_not_wait_out_the_queue_timeout(self):
        # `QueueExamplesIterable` gives up on a drained queue after 30 s, and nothing
        # used to post the sentinel that would end it sooner - so every run paid that
        # timeout, once per consumer round, after all its real work was done.
        runner = DynamicMultiprocessingRunner(
            num_workers=2,
            prefetch_factor=8,
            worker_init=SharedMock(),
            worker_finalize=SharedMock(),
            progress_report_interval=0.0,
            callback=CallbackManager([]),
        )
        ds = Dataset.from_dict({"obj": list(range(20))}).to_iterable_dataset(2)

        fn = SharedMock()
        start = time.perf_counter()
        runner.run(ds, fn)
        elapsed = time.perf_counter() - start

        fn.assert_has_calls([call(sample) for sample in ds], same_order=False)
        assert elapsed < 20, f"run took {elapsed:.1f}s, suggesting it sat out the queue timeout"


def _a_finalizer(batch):
    return None


def _another_finalizer(batch):
    return None


class TestFinalizerReuse:
    """The finalizer is re-sent only when it, or how it is called, actually changes."""

    @pytest.fixture
    def worker(self):
        queue = mp.Queue()
        worker = Worker(
            rank=0,
            num_workers=1,
            msg_queue=queue,
            progress_report_interval=0.0,
            worker_init=MagicMock(),
            worker_finalize=MagicMock(),
        )
        yield worker
        queue.close()

    def test_omits_a_finalizer_the_worker_already_has(self, worker):
        sent = []
        with patch.object(worker, "_send_ctx_bytes", side_effect=_capture_sent(sent)):
            worker.send_ctx(_standalone_ctx(finalizer=_a_finalizer, batch_size=8), blocking=False)
            worker.send_ctx(_standalone_ctx(finalizer=_a_finalizer, batch_size=8), blocking=False)

        assert sent[0].data_finalizer is not None
        assert sent[1].data_finalizer is None

    def test_resends_when_the_finalizer_changes(self, worker):
        sent = []
        with patch.object(worker, "_send_ctx_bytes", side_effect=_capture_sent(sent)):
            worker.send_ctx(_standalone_ctx(finalizer=_a_finalizer, batch_size=8), blocking=False)
            worker.send_ctx(
                _standalone_ctx(finalizer=_another_finalizer, batch_size=8), blocking=False
            )

        assert sent[1].data_finalizer is not None

    def test_resends_when_only_the_batch_size_changes(self, worker):
        # `apply_ctx` applies the batch size and formatting only alongside a non-None
        # finalizer, so dropping the finalizer would silently drop these with it - which
        # is exactly what separates a producer's context from a standalone's.
        sent = []
        with patch.object(worker, "_send_ctx_bytes", side_effect=_capture_sent(sent)):
            worker.send_ctx(_standalone_ctx(finalizer=_a_finalizer, batch_size=8), blocking=False)
            worker.send_ctx(_standalone_ctx(finalizer=_a_finalizer, batch_size=64), blocking=False)

        assert sent[1].data_finalizer is not None
        assert sent[1].data_finalizer_batch_size == 64


class TestContextPartCache:
    """One transform is pickled once for the whole run, not once per worker."""

    def test_reuses_the_same_buffer_for_the_same_object(self):
        cache = _ContextPartCache()
        first = cache.dumps(_a_transform)
        second = cache.dumps(_a_transform)

        assert first.payload is second.payload
        assert second.unwrap() is _a_transform

    def test_serialises_distinct_objects_separately(self):
        cache = _ContextPartCache()

        assert cache.dumps(_a_transform).payload != cache.dumps(_another_transform).payload

    def test_stops_growing_once_full(self):
        # Role switches build a fresh bound method every time, so entries must not
        # accumulate for the length of a run.
        cache = _ContextPartCache(max_entries=2)
        held = [lambda: None for _ in range(5)]
        for fn in held:
            cache.dumps(fn)

        assert len(cache._entries) == 2

    def test_workers_sharing_a_cache_pickle_a_transform_once(self):
        cache = _ContextPartCache()
        queue = mp.Queue()
        workers = [
            Worker(
                rank=rank,
                num_workers=2,
                msg_queue=queue,
                progress_report_interval=0.0,
                worker_init=MagicMock(),
                worker_finalize=MagicMock(),
                ctx_part_cache=cache,
            )
            for rank in range(2)
        ]
        try:
            sent = []
            for worker in workers:
                with patch.object(worker, "_send_ctx_bytes", side_effect=_capture_sent(sent)):
                    worker.send_ctx(_standalone_ctx(transform=_a_transform), blocking=False)

            # `_capture_sent` pickles what it captured, so buffer identity cannot
            # survive it; that the cache holds one entry is what says it pickled once.
            assert [type(ctx.data_transform) for ctx in sent] == [_PrePickled, _PrePickled]
            assert sent[0].data_transform.payload == sent[1].data_transform.payload
            assert len(cache._entries) == 1
        finally:
            queue.close()

    def test_a_received_context_carries_the_real_callables(self):
        # Whatever the wire format, what the worker applies has to be the originals.
        cache = _ContextPartCache()
        queue = mp.Queue()
        worker = Worker(
            rank=0,
            num_workers=1,
            msg_queue=queue,
            progress_report_interval=0.0,
            worker_init=MagicMock(),
            worker_finalize=MagicMock(),
            ctx_part_cache=cache,
        )
        try:
            worker.send_ctx(
                _standalone_ctx(transform=_a_transform, finalizer=_a_finalizer, batch_size=8),
                blocking=False,
            )
            received = worker._recv_ctx(blocking=True, timeout=5.0)
        finally:
            queue.close()

        assert received.data_transform is _a_transform
        assert received.data_finalizer is _a_finalizer
        assert received.data_finalizer_batch_size == 8


class TestSentinelDelivery:
    """A sentinel that could not be posted is owed, not forgotten."""

    def test_a_full_queue_leaves_the_sentinel_owed(self):
        # The queue holds one item per worker, so at the moment the end is detected it
        # can still be full of real data. Dropping the attempt there left consumers to
        # discover the end by timing out - the 30 s stall this whole mechanism exists to
        # remove, reappearing as a rare but severe collapse.
        controller = WorkerController(workers=[MagicMock()], prefetch=8, num_shards=1)
        controller.consumer_ranks = {0}
        controller.queue.put_nowait("data")  # maxsize is one worker, so now full

        assert controller.signal_no_more_data() == 0
        assert controller.consumers_awaiting_sentinel == {0}

        controller.queue.get_nowait()  # the consumer drains it

        assert controller.signal_no_more_data() == 1
        assert controller.consumers_awaiting_sentinel == set()
        assert controller.queue.get_nowait() is controller.queue_it.sentinel

    def test_a_consumer_is_not_sent_two_sentinels(self):
        controller = WorkerController(workers=[MagicMock(), MagicMock()], prefetch=8, num_shards=1)
        controller.consumer_ranks = {0}

        assert controller.signal_no_more_data() == 1
        assert controller.signal_no_more_data() == 0
        assert controller.queue.qsize() == 1

    def test_a_worker_re_entering_the_role_is_owed_another(self):
        # Freeing a worker and giving it the consumer role again is a fresh consumer
        # blocked on the queue, so the earlier sentinel does not cover it.
        controller = WorkerController(workers=[MagicMock()], prefetch=8, num_shards=1)
        controller.consumer_ranks = {0}
        controller.signal_no_more_data()
        controller.queue.get_nowait()

        controller.free_worker(0)
        controller.consumer_ranks = {0}

        assert controller.consumers_awaiting_sentinel == {0}
        assert controller.signal_no_more_data() == 1
