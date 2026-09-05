from unittest.mock import MagicMock, call, patch

import pytest
from datasets import Dataset

from crane.core.callbacks.base import CallbackManager
from crane.core.monitor import ProgressMonitor
from crane.core.runners.base import FailurePolicy, ShardProcessingError
from crane.core.runners.main_process_runner import MainProcessRunner
from crane.core.worker import reset_worker_info


class TestMainProcessRunner:
    @pytest.fixture
    def monitor(self, ds):
        monitor = ProgressMonitor(ds.n_shards, 1, None, 0)
        with patch(
            "crane.core.runners.main_process_runner.ProgressMonitor",
            MagicMock(return_value=monitor),
        ):
            yield monitor

    @pytest.fixture
    def ds(self):
        # create mock dataset
        samples = {"obj": [i for i in range(20)]}
        ds = Dataset.from_dict(samples)
        ds = ds.to_iterable_dataset(1)
        return ds

    @pytest.fixture(autouse=True)
    def reset_worker_info_after_test(self):
        """The worker info is process-global, so a run that fails before its own reset
        would otherwise leak into every following test."""
        yield
        reset_worker_info()

    @pytest.fixture
    def runner(self):
        return MainProcessRunner(
            batch_size=1,
            env_init=MagicMock(),
            env_finalize=MagicMock(),
            progress_report_interval=0.0,
            callback=CallbackManager([]),
        )

    def test_run(self, runner, ds, monitor):
        fn = MagicMock()
        runner.run(ds, fn)

        fn.assert_has_calls([call(sample) for sample in ds], any_order=True)
        assert monitor.num_processed_samples == 20

    def test_run_with_report_in_finally(self, runner, ds, monitor):
        runner._report_interval = float("inf")
        runner.run(ds, MagicMock())
        assert monitor.num_processed_samples == 20

    def test_run_with_keyboard_interrupt(self, runner, ds):
        fn = MagicMock(side_effect=KeyboardInterrupt)
        runner.run(ds, fn)

        fn.assert_called_once()

    def test_run_with_exception(self, runner, ds):
        # The workload raising is now a failed run rather than a logged line: the default
        # policy stops and raises, so a caller cannot mistake a partial dataset for a
        # complete one.
        fn = MagicMock(side_effect=RuntimeError)

        with pytest.raises(ShardProcessingError) as excinfo:
            runner.run(ds, fn)

        fn.assert_called_once()
        assert excinfo.value.failures[0].error_type == "RuntimeError"

    def test_run_with_exception_skipping_shards(self):
        # The old behaviour, now something the caller opts into: carry on past a bad
        # shard, and hear about it at the end. Three shards, so "carried on" is
        # distinguishable from "stopped at the first".
        ds = Dataset.from_dict({"obj": list(range(20))}).to_iterable_dataset(3)
        runner = MainProcessRunner(
            batch_size=8,
            env_init=MagicMock(),
            env_finalize=MagicMock(),
            progress_report_interval=0.0,
            callback=CallbackManager([]),
            failure_policy=FailurePolicy.SKIP_SHARD,
        )
        fn = MagicMock(side_effect=RuntimeError)

        with pytest.raises(ShardProcessingError) as excinfo:
            runner.run(ds, fn)

        # every shard was attempted, rather than stopping at the first
        assert len(excinfo.value.failures) == 3
        assert {f.shard_id for f in excinfo.value.failures} == {0, 1, 2}

    def test_failing_env_init_is_not_masked_by_unbound_local(self, runner, ds, monitor):
        # The finally block reports leftover samples, so it reads counters that used to be
        # assigned only after a successful initialization - hiding the actual failure.
        runner._env_init = MagicMock(side_effect=RuntimeError("env init failed"))

        with pytest.raises(BaseException) as exc_info:
            runner.run(ds, MagicMock())

        assert not isinstance(exc_info.value, UnboundLocalError)
