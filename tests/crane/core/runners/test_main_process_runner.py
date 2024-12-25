from unittest.mock import MagicMock, call, patch

import pytest
from datasets import Dataset

from crane.core.callbacks.base import CallbackManager
from crane.core.monitor import ProgressMonitor
from crane.core.runners.main_process_runner import MainProcessRunner


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

    @pytest.fixture
    def runner(self):
        return MainProcessRunner(
            batch_size=1,
            initialize=MagicMock(),
            finalize=MagicMock(),
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
        fn = MagicMock(side_effect=RuntimeError)
        runner.run(ds, fn)

        fn.assert_called_once()
