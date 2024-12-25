from unittest.mock import ANY, MagicMock, patch

import pytest

from crane.core.callbacks.tqdm_reporter import _run_tqdm
from crane.core.monitor import ProgressMonitor
from crane.core.runners.base import WorkerRole


@pytest.fixture
def monitor():
    """Create a mock ProgressMonitor."""
    monitor = MagicMock(spec=ProgressMonitor)
    monitor._done = MagicMock()
    monitor.num_processed_samples = 0
    monitor.completed_shards = []
    monitor.num_workers = 5
    monitor._num_shards = 10
    monitor._roles = [WorkerRole.PRODUCER, WorkerRole.CONSUMER, WorkerRole.STANDALONE]
    monitor.alive_workers = [1, 2, 3]
    return monitor


@patch("crane.core.callbacks.tqdm_reporter.tqdm")
def test_run_tqdm_progress_updates(mock_tqdm, monitor):
    """Test that the _run_tqdm function updates the progress bar correctly."""
    # Setup mock tqdm
    mock_pbar_instance = MagicMock()
    mock_tqdm.return_value.__enter__.return_value = mock_pbar_instance
    # Simulate the monitor returning values over time
    monitor.num_processed_samples = 50
    monitor.completed_shards = [1, 2, 3]
    monitor._done.wait.side_effect = [False, False, True]  # Tracker runs twice, then finishes
    monitor.samples_per_second = 0.0
    # Run the function
    _run_tqdm(monitor, update_interval=0.1)
    # Assertions for tqdm calls
    assert mock_tqdm.call_count == 1
    mock_tqdm.assert_called_once_with(total=monitor._num_shards, desc=ANY, unit="sh", **{})
    # Ensure progress bar updates happen
    assert mock_pbar_instance.update.call_count == 3
    assert mock_pbar_instance.set_description.call_count == 3
    assert mock_pbar_instance.set_postfix_str.call_count == 3
