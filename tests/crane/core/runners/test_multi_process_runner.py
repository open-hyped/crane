import json
import multiprocessing as mp
import time
from queue import Full, Queue
from unittest.mock import MagicMock, call, patch

import datasets
import dill
import pytest
from datasets import Dataset
from datasets.iterable_dataset import SelectColumnsIterable

from crane.core.callbacks.base import Callback, CallbackManager
from crane.core.runners.base import (
    FailurePolicy,
    ShardProcessingError,
    WorkerProcessingStage,
    WorkerRole,
)
from crane.core.runners.multi_process_runner import (
    Command,
    CommandType,
    ConsumerProducerBalancer,
    DynamicMultiprocessingRunner,
    MessageType,
    Source,
    Worker,
    WorkerContext,
    WorkerController,
    WorkerSetup,
    _CountedQueue,
)
from tests.third_party.sharedmock.mock import SharedMock


def _assign_shard(shard_id: int = 0) -> Command:
    return Command(CommandType.ASSIGN, role=WorkerRole.STANDALONE, shard_id=shard_id)


def _message_types(msg_queue) -> list[int]:
    """Drain whatever the worker has reported so far."""
    types = []
    while True:
        try:
            types.append(json.loads(msg_queue.get(timeout=0.5).decode("utf-8"))["type"])
        except Exception:
            return types


def _setup(transform, finalizer, data_source=None, num_shards=1):
    return WorkerSetup(
        transform=transform,
        finalizer=finalizer,
        finalizer_batch_size=None,
        finalizer_formatting=None,
        data_source=data_source,
        num_shards=num_shards,
    )


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
    def finalizer(self):
        return MagicMock(return_value=None)

    @pytest.fixture
    def dataset(self):
        dummy_data = [
            {"a": 0, "b": [1, 2, 3, 4]},
            {"a": 1, "b": [5, 6]},
            {"a": 1, "b": [7, 8, 9, 10]},
        ]
        return Dataset.from_list(dummy_data).to_iterable_dataset(1)

    @pytest.fixture
    def data_stream(self, dataset):
        return dataset._ex_iterable

    @pytest.fixture
    def worker(self, msg_queue, worker_init, worker_finalizer, transform, finalizer, data_stream):
        worker = Worker(
            rank=0,
            num_workers=1,
            msg_queue=msg_queue,
            progress_report_interval=0.0,
            worker_init=worker_init,
            worker_finalize=worker_finalizer,
            setup=_setup(transform, finalizer, data_source=data_stream),
        )
        worker._build_contexts()
        return worker

    @pytest.fixture
    def busy_worker(self, worker, data_stream):
        """A worker part-way through a stream, as far as command acceptance is concerned."""
        worker._accept(_assign_shard(), busy=False)
        return worker

    def test_builds_one_context_per_role(self, worker, transform, finalizer):
        assert worker._standalone_ctx.role is WorkerRole.STANDALONE
        assert worker._consumer_ctx.role is WorkerRole.CONSUMER
        assert worker._producer_ctx.role is WorkerRole.PRODUCER

        # standalone and consumer run the same pipeline, and differ in where they read
        assert worker._standalone_ctx.transform is transform
        assert worker._consumer_ctx.transform is transform
        assert worker._standalone_ctx.source is Source.SHARD
        assert worker._consumer_ctx.source is Source.QUEUE
        # a producer keeps whatever stream it was promoted from
        assert worker._producer_ctx.source is Source.CURRENT

    def test_request_work_applies_an_assign(self, msg_queue, worker):
        worker._recv_resp_conn.recv = MagicMock()
        worker.send_command(_assign_shard())

        worker._request_work()

        msg = json.loads(msg_queue.get(timeout=1.0).decode("utf-8"))
        assert msg["rank"] == worker._rank
        assert msg["type"] == MessageType.CTX_REQUEST.value
        assert worker._active_ctx is worker._standalone_ctx
        assert worker._stream is not None

    def test_an_idle_worker_refuses_a_switch(self, worker):
        # A switch keeps the stream the worker is working through; an idle worker has
        # none, so refusing tells the controller to try another rank.
        assert not worker._accept(Command(CommandType.SWITCH, role=WorkerRole.PRODUCER), busy=False)
        assert worker._active_ctx is None

    def test_a_busy_worker_refuses_an_assign(self, busy_worker):
        # An assign brings a new stream, which would discard the one in progress.
        assert not busy_worker._accept(_assign_shard(), busy=True)
        assert busy_worker._active_ctx is busy_worker._standalone_ctx

    def test_a_switch_keeps_the_stream(self, busy_worker):
        stream = busy_worker._stream

        assert busy_worker._accept(Command(CommandType.SWITCH, role=WorkerRole.PRODUCER), busy=True)

        assert busy_worker._active_ctx is busy_worker._producer_ctx
        assert busy_worker._stream is stream, "a producer carries on with its own shard"

    def test_a_demoted_producer_resumes_its_shard(self, busy_worker):
        busy_worker._accept(Command(CommandType.SWITCH, role=WorkerRole.PRODUCER), busy=True)
        stream = busy_worker._stream

        assert busy_worker._accept(
            Command(CommandType.SWITCH, role=WorkerRole.STANDALONE), busy=True
        )

        assert busy_worker._active_ctx is busy_worker._standalone_ctx
        assert busy_worker._stream is stream

    def test_stop_is_accepted_in_any_state(self, worker, busy_worker):
        assert worker._accept(Command(CommandType.STOP), busy=False)
        assert worker._stop

    def test_send_command_survives_full_queue(self, worker):
        # A multiprocessing queue is fed by a background thread, so an item can land in
        # the slot between the drain and the put. That used to raise `Full` out of the
        # send and kill the controller's message loop, leaving workers waiting for a
        # command that never arrived - the run then hung instead of finishing.
        command = _assign_shard()
        worker._recv_resp_conn.recv = MagicMock()

        real_put_nowait = worker._command_queue.put_nowait
        calls = []

        def flaky_put_nowait(item):
            calls.append(item)
            if len(calls) == 1:
                raise Full()
            real_put_nowait(item)

        worker._command_queue.put_nowait = flaky_put_nowait
        worker.send_command(command)

        assert len(calls) == 2, "should have retried after the queue reported itself full"
        assert worker._command_queue.get(timeout=1.0) == command

    def test_send_command_gives_up_without_raising(self, worker):
        # A worker that never drains its queue must not take the controller down with it.
        worker._recv_resp_conn.recv = MagicMock()
        worker._command_queue.put_nowait = MagicMock(side_effect=Full())

        worker.send_command(_assign_shard())  # must return rather than raise

        assert worker._command_queue.put_nowait.call_count > 1

    @patch("crane.core.runners.multi_process_runner.set_worker_info")
    def test_run(self, mock_set_worker_info, worker, data_stream, transform, finalizer):
        worker._request_work = MagicMock(side_effect=[True, False].pop)
        worker._accept(_assign_shard(), busy=False)

        worker.run()

        mock_set_worker_info.assert_called_once()
        transform.assert_called_once()
        finalizer.assert_has_calls([call(x) for _, x in data_stream], any_order=True)
        worker._worker_init.assert_called_once()
        worker._worker_finalize.assert_called_once()

    @patch("crane.core.runners.multi_process_runner.set_worker_info")
    def test_run_with_role_switch(
        self, mock_set_worker_info, msg_queue, worker, data_stream, transform
    ):
        worker._request_work = MagicMock(side_effect=[True, False].pop)
        worker._accept(_assign_shard(), busy=False)
        # Both ends, or replacing only the reader drops the last reference to it, the
        # pipe is collected, and the worker's reply raises BrokenPipeError. The handshake
        # is not what this test is about.
        worker._recv_resp_conn = MagicMock()
        worker._recv_resp_conn.poll = MagicMock(return_value=True)
        worker._send_resp_conn = MagicMock()
        worker._recv_command = MagicMock(
            return_value=Command(CommandType.SWITCH, role=WorkerRole.STANDALONE)
        )

        worker.run()

        mock_set_worker_info.assert_called_once()
        transform.assert_called()
        # What matters is that the switch was taken and announced, not how many times the
        # loop happened to poll for one.
        assert worker._recv_command.called
        assert worker._active_ctx is worker._standalone_ctx
        assert MessageType.CTX_SWITCH.value in _message_types(msg_queue)
        worker._worker_init.assert_called_once()
        worker._worker_finalize.assert_called_once()

    @patch("crane.core.runners.multi_process_runner.set_worker_info")
    def test_run_with_abort(self, mock_set_worker_info, worker, transform, finalizer):
        worker._request_work = MagicMock(side_effect=[True, False].pop)
        worker._accept(_assign_shard(), busy=False)
        worker._recv_command = MagicMock(return_value=Command(CommandType.STOP))

        worker.run()

        transform.assert_called_once()
        mock_set_worker_info.assert_called_once()
        assert len(worker._recv_command.mock_calls) == 1
        finalizer.assert_has_calls([call({"a": 0, "b": [1, 2, 3, 4]})], any_order=True)


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


class TestConsumerShutdown:
    """Consumers are told the data ran out instead of inferring it from a timeout."""

    def test_close_stream_sets_the_shared_flag(self):
        controller = WorkerController(
            workers=[MagicMock()], prefetch=8, num_shards=1, queue=Queue(maxsize=1)
        )
        assert not controller.stream_closed.is_set()

        controller.close_stream()

        assert controller.stream_closed.is_set()

    def test_close_stream_is_idempotent(self):
        # It is called from several points in the message loop, whenever the condition
        # happens to hold, so calling it repeatedly must be free of consequence.
        controller = WorkerController(
            workers=[MagicMock()], prefetch=8, num_shards=1, queue=Queue(maxsize=1)
        )
        controller.close_stream()
        controller.close_stream()

        assert controller.stream_closed.is_set()

    def test_run_does_not_wait_out_the_queue_timeout(self):
        # `QueueExamplesIterable` gives up on a queue delivering nothing after 30 s, and
        # nothing used to mark the stream closed sooner - so every run paid that timeout
        # after all its real work was done.
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


class TestCountedQueue:
    """The queue replacing the manager-backed one, and the count that made it possible."""

    def test_tracks_its_own_length(self):
        # `mp.Queue.qsize()` is backed by `sem_getvalue()`, which macOS does not
        # implement - the reason a manager proxy was used at all. Counting puts and gets
        # gives a length everywhere, so a plain queue and its single pipe can be kept.
        queue = _CountedQueue(maxsize=4)
        assert queue.empty() and queue.qsize() == 0

        queue.put("a")
        queue.put("b")
        assert queue.qsize() == 2 and not queue.empty()

        assert queue.get(timeout=5) == "a"
        assert queue.qsize() == 1
        assert queue.get(timeout=5) == "b"
        assert queue.empty()

    def test_reports_full_without_blocking_forever(self):
        queue = _CountedQueue(maxsize=1)
        queue.put("a")
        with pytest.raises(Full):
            queue.put("b", timeout=0.1)
        assert queue.get(timeout=5) == "a"


def _raise_on_everything(x):
    raise RuntimeError("workload failed")


def _raise_on_one(x):
    # Fails a single value, so the rest of the dataset is still processable.
    if x["obj"] == 7:
        raise RuntimeError("workload failed")
    return x


class TestWorkerFailures:
    """A workload that raises must not look like a run that succeeded."""

    def _runner(self, policy, callbacks=()):
        return DynamicMultiprocessingRunner(
            num_workers=2,
            prefetch_factor=8,
            worker_init=SharedMock(),
            worker_finalize=SharedMock(),
            progress_report_interval=0.0,
            callback=CallbackManager(list(callbacks)),
            failure_policy=policy,
        )

    @pytest.fixture
    def ds(self):
        return Dataset.from_dict({"obj": list(range(20))}).to_iterable_dataset(2)

    def test_fail_fast_raises(self, ds):
        # Before this, every worker logged its own traceback to its own stderr and the
        # run reported success having written nothing.
        runner = self._runner(FailurePolicy.FAIL_FAST)

        with pytest.raises(ShardProcessingError) as excinfo:
            runner.run(ds.map(_raise_on_everything), SharedMock())

        assert excinfo.value.failures
        assert excinfo.value.failures[0].error_type == "RuntimeError"
        assert "workload failed" in excinfo.value.failures[0].stack_trace

    def test_fail_fast_is_the_default(self, ds):
        runner = DynamicMultiprocessingRunner(
            num_workers=2,
            prefetch_factor=8,
            worker_init=SharedMock(),
            worker_finalize=SharedMock(),
            progress_report_interval=0.0,
            callback=CallbackManager([]),
        )

        with pytest.raises(ShardProcessingError):
            runner.run(ds.map(_raise_on_everything), SharedMock())

    def test_skip_shard_keeps_the_rest_and_still_reports(self, ds):
        runner = self._runner(FailurePolicy.SKIP_SHARD)
        fn = SharedMock()

        with pytest.raises(ShardProcessingError) as excinfo:
            runner.run(ds.map(_raise_on_one), fn)

        # the shard that did not contain the bad value was written
        assert fn.call_count > 0
        assert len(excinfo.value.failures) >= 1

    def test_a_failure_reaches_the_callback(self, ds):
        callback = MagicMock(wraps=Callback())
        runner = self._runner(FailurePolicy.FAIL_FAST, callbacks=[callback])

        with pytest.raises(ShardProcessingError):
            runner.run(ds.map(_raise_on_everything), SharedMock())

        assert callback.on_exception.called
        failure = callback.on_exception.call_args[0][1]
        assert failure.error_type == "RuntimeError"

    def test_a_failing_shard_is_abandoned_not_retried(self, ds):
        # The worker used to leave `stream_exhausted` false after reporting, so it ran the
        # same failing context again, forever. A handful of failures is a shard being
        # given up on; hundreds is that loop.
        runner = self._runner(FailurePolicy.SKIP_SHARD)

        with pytest.raises(ShardProcessingError) as excinfo:
            runner.run(ds.map(_raise_on_everything), SharedMock())

        assert len(excinfo.value.failures) <= 8, "the same shard is being retried"
