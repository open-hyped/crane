import unittest
from unittest.mock import ANY, MagicMock, patch

from datasets import IterableDataset

from crane.core.consumer import DatasetConsumer


class TestDatasetConsumer(unittest.TestCase):
    @patch("crane.core.consumer.MainProcessRunner")
    @patch("crane.core.consumer.DynamicMultiprocessingRunner")
    def test_consume_single_process(self, mock_dynamic_runner, mock_main_runner):
        # Create a mock dataset
        mock_dataset = MagicMock(spec=IterableDataset)

        # create mock finalizer setup
        mock_finalizer = MagicMock()
        mock_batch_size = MagicMock()
        mock_formatting = MagicMock()

        # Mock initialization and finalization functions
        mock_initialize = MagicMock()
        mock_finalize = MagicMock()

        # Create the dataset consumer with a single process
        consumer = DatasetConsumer(
            num_proc=1,  # Single process mode
            on_start=mock_initialize,
            on_finish=mock_finalize,
            disable_tqdm=False,
        )

        # Mock the runner
        mock_runner_instance = mock_main_runner.return_value
        mock_runner_instance.run = MagicMock()

        # Call the consumer
        consumer.consume(mock_dataset, mock_finalizer, mock_batch_size, mock_formatting)

        # Ensure the MainProcessRunner was created
        mock_main_runner.assert_called_once_with(
            batch_size=consumer._prefetch,
            env_init=mock_initialize,
            env_finalize=mock_finalize,
            progress_report_interval=consumer._report_interval,
            callback=consumer._callback,
        )

        # Ensure the run method was called with the dataset and function
        mock_runner_instance.run.assert_called_once_with(
            mock_dataset, mock_finalizer, mock_batch_size, mock_formatting
        )

        # Ensure the DynamicMultiprocessingRunner was not called
        mock_dynamic_runner.assert_not_called()

    @patch("crane.core.consumer.MainProcessRunner")
    @patch("crane.core.consumer.DynamicMultiprocessingRunner")
    def test_consume_multi_process(self, mock_dynamic_runner, mock_main_runner):
        # Create a mock dataset
        mock_dataset = MagicMock(spec=IterableDataset)

        # create mock finalizer setup
        mock_finalizer = MagicMock()
        mock_batch_size = MagicMock()
        mock_formatting = MagicMock()

        # Mock initialization and finalization functions
        mock_initialize = MagicMock()
        mock_finalize = MagicMock()

        # Create the dataset consumer with multiple processes
        consumer = DatasetConsumer(
            num_proc=4,  # Multi-process mode
            on_start=mock_initialize,
            on_finish=mock_finalize,
            disable_tqdm=False,
        )

        # Mock the runner
        mock_runner_instance = mock_dynamic_runner.return_value
        mock_runner_instance.run = MagicMock()

        # Call the consumer
        consumer.consume(mock_dataset, mock_finalizer, mock_batch_size, mock_formatting)

        # Ensure the DynamicMultiprocessingRunner was created
        mock_dynamic_runner.assert_called_once_with(
            num_workers=consumer._num_proc,
            prefetch_factor=consumer._prefetch,
            worker_init=mock_initialize,
            worker_finalize=mock_finalize,
            progress_report_interval=consumer._report_interval,
            callback=consumer._callback,
        )

        # Ensure the run method was called with the dataset and function
        mock_runner_instance.run.assert_called_once_with(
            mock_dataset, mock_finalizer, mock_batch_size, mock_formatting
        )

        # Ensure the MainProcessRunner was not called
        mock_main_runner.assert_not_called()

    @patch("crane.core.consumer.TqdmReporterCallback")
    def test_tqdm_callback_added(self, tqdm_reporter_cls):
        # Create the dataset consumer with disable_tqdm=False (tqdm should be enabled)
        consumer = DatasetConsumer(
            num_proc=1,  # Single process mode
            disable_tqdm=True,  # Tqdm should be enabled
        )

        # Ensure TqdmReporterCallback was added to the callback list
        assert all(cb != tqdm_reporter_cls() for cb in consumer._callback._callbacks)

        # Create the dataset consumer with disable_tqdm=False (tqdm should be enabled)
        consumer = DatasetConsumer(
            num_proc=1,  # Single process mode
            disable_tqdm=False,  # Tqdm should be enabled
        )

        # Ensure TqdmReporterCallback was added to the callback list
        assert any(cb == tqdm_reporter_cls() for cb in consumer._callback._callbacks)
