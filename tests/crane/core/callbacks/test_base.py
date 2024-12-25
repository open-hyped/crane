from unittest.mock import MagicMock

import pytest
from datasets import IterableDataset

from crane.core.callbacks.base import CallbackManager
from crane.core.monitor import ProgressMonitor


@pytest.fixture
def mock_callback():
    """Create a mock callback."""
    return MagicMock()


@pytest.fixture
def callback_manager(mock_callback):
    """Initialize the CallbackManager with the mock callback."""
    return CallbackManager(callbacks=[mock_callback])


@pytest.fixture
def monitor():
    """Create a mock ProgressMonitor."""
    return MagicMock(spec=ProgressMonitor)


@pytest.fixture
def dataset():
    """Create a mock IterableDataset."""
    return MagicMock(spec=IterableDataset)


def test_on_start_called(callback_manager, mock_callback, monitor, dataset):
    """Test that on_start is called on all callbacks."""
    callback_manager.on_start(monitor, dataset)
    mock_callback.on_start.assert_called_once_with(monitor, dataset)


def test_on_shard_in_progress_called(callback_manager, mock_callback, monitor):
    """Test that on_shard_in_progress is called on all callbacks."""
    shard_id = 1
    callback_manager.on_shard_in_progress(monitor, shard_id)
    mock_callback.on_shard_in_progress.assert_called_once_with(monitor, shard_id)


def test_on_shard_completed_called(callback_manager, mock_callback, monitor):
    """Test that on_shard_completed is called on all callbacks."""
    shard_id = 1
    callback_manager.on_shard_completed(monitor, shard_id)
    mock_callback.on_shard_completed.assert_called_once_with(monitor, shard_id)


def test_on_shard_canceled_called(callback_manager, mock_callback, monitor):
    """Test that on_shard_canceled is called on all callbacks."""
    shard_id = 1
    callback_manager.on_shard_canceled(monitor, shard_id)
    mock_callback.on_shard_canceled.assert_called_once_with(monitor, shard_id)


def test_on_stopping_called(callback_manager, mock_callback, monitor):
    """Test that on_stopping is called on all callbacks."""
    callback_manager.on_stopping(monitor)
    mock_callback.on_stopping.assert_called_once_with(monitor)


def test_on_done_called(callback_manager, mock_callback, monitor):
    """Test that on_done is called on all callbacks."""
    callback_manager.on_done(monitor)
    mock_callback.on_done.assert_called_once_with(monitor)
