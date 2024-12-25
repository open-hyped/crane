import json
import multiprocessing as mp
from unittest.mock import MagicMock, call, patch

import pytest
from datasets import Dataset
from sharedmock.mock import SharedMock

from crane.core.callbacks.base import CallbackManager
from crane.core.runners.base import WorkerProcessingStage, WorkerRole
from crane.core.runners.multi_process_runner import (
    ConsumerProducerBalancer,
    DynamicMultiprocessingRunner,
    MessageType,
    Serializer,
    Worker,
    WorkerContext,
)


class TestWorker:
    @pytest.fixture
    def worker_init(self):
        return MagicMock()

    @pytest.fixture
    def worker_finalizer(self):
        return MagicMock()

    @pytest.fixture
    def msg_pipe(self):
        return mp.Pipe(duplex=False)

    @pytest.fixture
    def transform(self):
        return MagicMock()

    @pytest.fixture
    def context(self, transform):
        return WorkerContext(
            role=WorkerRole.STANDALONE,
            data_stream=[MagicMock(), MagicMock(), MagicMock()],
            data_transform=lambda x: map(transform, x),
            data_finalizer=MagicMock(),
            stop=False,
        )

    @pytest.fixture
    def worker(self, msg_pipe, worker_init, worker_finalizer, context):
        _, send_msg_conn = msg_pipe

        worker = Worker(
            rank=0,
            num_workers=1,
            send_msg_conn=send_msg_conn,
            progress_report_interval=0.0,
            worker_init=worker_init,
            worker_finalize=worker_finalizer,
        )
        worker._ctx = context
        return worker

    def test_request_new_ctx(self, msg_pipe, worker):
        recv_msg, _ = msg_pipe

        ctx = WorkerContext(
            role=WorkerRole.STANDALONE,
            data_stream="STREAM",
            data_transform="TRANSFORM",
            data_finalizer="FINALIZE",
            stop=False,
        )
        # mock context conn receiver to avoid deadlock
        worker._parent_ctx_conn.recv = MagicMock()
        # send new context before worker request to avoid deadlock in worker
        worker.send_ctx(ctx)
        worker._parent_ctx_conn.recv.assert_called_once()

        # request new context
        worker._request_new_ctx()

        # make sure the worker send a request context message
        assert recv_msg.poll(timeout=0.1)
        msg = json.loads(recv_msg.recv_bytes().decode("utf-8"))
        assert msg["rank"] == worker._rank
        assert msg["type"] == MessageType.CTX_REQUEST.value

        # check if new context was applied
        assert worker._ctx == ctx

    @patch("crane.core.runners.multi_process_runner.set_worker_info")
    def test_run(self, mock_set_worker_info, worker, transform):
        # mock request new context
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        # run worker
        worker.run()

        mock_set_worker_info.assert_called_once()
        # make sure all samples have been processed
        worker._ctx.data_finalizer.assert_has_calls(
            [call(transform(x)) for x in worker._ctx.data_stream], any_order=True
        )
        worker._worker_init.assert_called_once()
        worker._worker_finalize.assert_called_once()

    @patch("crane.core.runners.multi_process_runner.set_worker_info")
    def test_run_with_ctx_update(self, mock_set_worker_info, worker, transform):
        # mock request new and check context
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        # mock context connection
        worker._child_ctx_conn = MagicMock()
        worker._child_ctx_conn.poll = MagicMock(return_value=True)
        # mock receive context function
        ctx_update = WorkerContext(role=WorkerRole.STANDALONE, stop=False)
        worker._recv_ctx = MagicMock(return_value=ctx_update)

        worker.run()

        mock_set_worker_info.assert_called_once()
        # make sure all samples have been processed
        assert len(worker._recv_ctx.mock_calls) == 3
        worker._ctx.data_finalizer.assert_has_calls(
            [call(transform(x)) for x in worker._ctx.data_stream], any_order=True
        )
        worker._worker_init.assert_called_once()
        worker._worker_finalize.assert_called_once()

    @patch("crane.core.runners.multi_process_runner.set_worker_info")
    def test_run_with_abort(self, mock_set_worker_info, worker, transform):
        # mock request new and check context
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        # mock context connection
        worker._child_ctx_conn = MagicMock()
        worker._child_ctx_conn.poll = MagicMock(return_value=True)
        # mock receive context function
        ctx_update = WorkerContext(role=WorkerRole.STANDALONE, stop=True)
        worker._recv_ctx = MagicMock(return_value=ctx_update)

        worker.run()

        mock_set_worker_info.assert_called_once()
        # make sure all samples have been processed
        assert len(worker._recv_ctx.mock_calls) == 1
        worker._ctx.data_finalizer.assert_has_calls(
            [call(transform(worker._ctx.data_stream[0]))], any_order=True
        )

    @patch("crane.core.runners.multi_process_runner.set_worker_info")
    def test_error_logging(self, mock_set_worker_info, worker):
        # mock finalizer to throw runtime error
        worker._ctx.data_finalizer = MagicMock(side_effect=RuntimeError)
        # mock request new and check context
        worker._logger = MagicMock()
        worker._logger.error = MagicMock()
        # mock request new context and run worker
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        worker.run()
        # make sure errors were logged
        assert len(worker._logger.error.mock_calls) == 3

        worker._logger.error.reset_mock()
        worker._worker_init = MagicMock(side_effect=RuntimeError)
        # mock request new context and run worker
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        worker.run()
        # make sure errors were logged
        worker._logger.error.assert_called_once()

        worker._logger.error.reset_mock()
        worker._worker_finalize = MagicMock(side_effect=RuntimeError)
        # mock request new context and run worker
        worker._request_new_ctx = MagicMock(side_effect=[True, False].pop)
        worker.run()
        # make sure errors were logged
        assert len(worker._logger.error.mock_calls) == 2


class TestSerializer:
    def test_serialization_deserialization(self):
        serializer = Serializer(batch_size=5)
        samples = [{"key": i} for i in range(32)]

        # Serialize and deserialize the samples
        serialized = list(serializer.serialize(samples))
        deserialized = list(serializer.deserialize(serialized))

        # Check that the deserialized output matches the original samples
        assert deserialized == samples, "Deserialized output does not match the original samples"


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
        mock_monitor.elapsed_time_averages.side_effect = lambda _: {
            WorkerProcessingStage.FINALIZE: 1.0,  # Producer block time
            WorkerProcessingStage.STREAM: 1.0,  # Consumer block time
        }

        action = balancer.callback()
        assert action == ConsumerProducerBalancer.Action.NO_ACTION


def _double_fn(x):
    return {"obj": x["obj"] * 2}


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

    @pytest.mark.parametrize("num_shards", [1, 2, 3])
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

    def test_run_with_keyboard_interrupt(self, runner, ds):
        def raise_exc(sample):
            raise KeyboardInterrupt()

        runner.run(ds, raise_exc)

    @pytest.mark.parametrize("nested", [0, 1, 2])
    def test_prepare_dataset(self, nested, runner, ds):
        # apply map function
        mapped_ds = ds
        for _ in range(nested):
            mapped_ds = mapped_ds.map(_double_fn)

        src_ex_it, pipeline = runner._prepare_dataset(mapped_ds)

        # check output
        assert isinstance(src_ex_it, type(ds._ex_iterable))

        # test pipeline output
        expected = list(pipeline(src_ex_it))
        actual = list(mapped_ds)
        assert actual == expected
