import json
import os
from copy import deepcopy
from dataclasses import dataclass
from unittest.mock import ANY, MagicMock, call, patch

import datasets
import dill
import pyarrow as pa
import pytest
from datasets import Dataset, IterableDataset, IterableDatasetDict

from crane.core.batching import BatchBuffer
from crane.core.naming import ShardName
from crane.core.utils import chdir
from crane.core.worker import get_worker_info, reset_worker_info, set_worker_info
from crane.core.writer import BaseDatasetWriter


class TestBaseDatasetWriter:
    def test_supported_write_formats(self) -> None:
        with pytest.raises(TypeError):
            # no write format supported
            class MockDatasetWriter(BaseDatasetWriter):
                SHARD_FILE_EXTENSION = "mock"
                initialize = MagicMock()
                finalize = MagicMock()
                initialize_shard = MagicMock()
                finalize_shard = MagicMock()

        with pytest.raises(TypeError, match="SHARD_FILE_EXTENSION"):
            # the base class names every shard, so a writer has to say what to call it
            class MockDatasetWriter(BaseDatasetWriter):
                write_batch_py = MagicMock()
                initialize = MagicMock()
                finalize = MagicMock()
                initialize_shard = MagicMock()
                finalize_shard = MagicMock()

        class MockDatasetWriter(BaseDatasetWriter):
            SHARD_FILE_EXTENSION = "mock"
            write_batch_py = MagicMock()
            initialize = MagicMock()
            finalize = MagicMock()
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        assert "python" in MockDatasetWriter.SUPPORTED_FORMATS
        assert len(MockDatasetWriter.SUPPORTED_FORMATS) == 1

        class MockDatasetWriter(BaseDatasetWriter):
            SHARD_FILE_EXTENSION = "mock"
            write_batch_arrow = MagicMock()
            initialize = MagicMock()
            finalize = MagicMock()
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        assert "arrow" in MockDatasetWriter.SUPPORTED_FORMATS
        assert len(MockDatasetWriter.SUPPORTED_FORMATS) == 1

        class MockDatasetWriter(BaseDatasetWriter):
            SHARD_FILE_EXTENSION = "mock"
            write_batch_py = MagicMock()
            write_batch_arrow = MagicMock()
            initialize = MagicMock()
            finalize = MagicMock()
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        assert "arrow" in MockDatasetWriter.SUPPORTED_FORMATS
        assert "python" in MockDatasetWriter.SUPPORTED_FORMATS
        assert len(MockDatasetWriter.SUPPORTED_FORMATS) == 2

    def test_get_write_fn(self) -> None:
        class MockDatasetWriter(BaseDatasetWriter):
            SHARD_FILE_EXTENSION = "mock"
            write_batch_py = MagicMock()
            write_batch_arrow = MagicMock()
            initialize = MagicMock()
            finalize = MagicMock()
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        MockDatasetWriter.write_batch_py.__get__ = MagicMock()
        MockDatasetWriter.write_batch_arrow.__get__ = MagicMock()

        mock_writer = MockDatasetWriter("out")
        ds = IterableDataset.from_generator([])

        # None type is python
        formatting, writer_fn = mock_writer._get_write_fn(ds.with_format(type=None))
        assert formatting == "python"
        assert writer_fn == mock_writer.write_batch_py
        # python
        formatting, writer_fn = mock_writer._get_write_fn(ds.with_format(type="python"))
        assert formatting == "python"
        assert writer_fn == mock_writer.write_batch_py
        # arrow
        formatting, writer_fn = mock_writer._get_write_fn(ds.with_format(type="arrow"))
        assert formatting == "arrow"
        assert writer_fn == mock_writer.write_batch_arrow

    @pytest.mark.parametrize("with_sharding", [True, False])
    def test_write_split(self, tmp_path, with_sharding):
        ds = Dataset.from_dict({"obj": [0]})
        ds = ds.to_iterable_dataset(1)

        class MockDatasetWriter(BaseDatasetWriter):
            SHARD_FILE_EXTENSION = "mock"
            write_batch_py = MagicMock()
            initialize = MagicMock()
            finalize = MagicMock()
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        MockDatasetWriter.write_batch_py.__get__ = MagicMock()
        MockDatasetWriter.write_batch_py.__qualname__ = "MockDatasetWriter.write_batch_py"

        with (
            patch("crane.core.writer.DatasetConsumer") as consumer_mock,
            patch("crane.core.writer.ShardingController") as sharding_mock,
        ):
            sharding_mock().is_active = with_sharding
            sharding_mock.reset_mock()

            writer = MockDatasetWriter(
                save_dir=tmp_path, overwrite=True, write_batch_size=1, num_proc=1
            )
            # the steps `write` takes for one split, which is what this exercises
            split = writer._prepare_split(ds)
            writer._consume_dataset(split, save_dir=tmp_path)
            writer._finalize_split(split, save_dir=tmp_path)

            # check sharding strategy
            sharding_mock.assert_called_once()

            # check dataset consumer
            consumer_mock.assert_called_once()
            consumer_mock().consume.assert_called_once_with(
                ds, finalizer=ANY, batch_size=1, formatting="python"
            )

            # get arguments to consumer initializer
            fn = consumer_mock().consume.mock_calls[0].kwargs["finalizer"]
            init = consumer_mock.mock_calls[0].kwargs["on_start"]
            finalize = consumer_mock.mock_calls[0].kwargs["on_finish"]

            # Apply the function. A real batch rather than a mock: the writer regroups what
            # it is handed into batches of `write_batch_size`, so it has to be able to count
            # the rows. One row against a size of one passes straight through.
            #
            # Inside a worker, because that is where the write path runs and where the
            # buffer lives - the sharding controller is mocked here, so the shard that would
            # have created it never opens.
            reset_worker_info()
            set_worker_info(rank=0, num_workers=1, seed=None)
            get_worker_info().ctx.buffer = BatchBuffer(1)
            get_worker_info().ctx.write_fn = None
            mock_batch = {"obj": [0]}
            sharding_mock().callback.return_value = mock_batch
            sharding_mock.reset_mock(return_value=False)
            sharding_mock().callback.return_value = mock_batch
            fn(mock_batch)
            # make sure that the callback is called first, then the write sample
            # and finally the update function. The callback also opens the shard for the
            # batch, so it runs for every strategy, including NONE.
            sharding_mock().callback.assert_called_once_with(mock_batch)
            writer.write_batch_py.assert_called_once_with(mock_batch)
            sharding_mock().update(writer.write_batch_py())

            # starting a worker must not open a shard - that only happens once a batch
            # reaches the worker, so a worker without data leaves no empty shard behind
            init()
            sharding_mock().initialize.assert_not_called()
            writer.initialize.assert_called_once()

            # make sure the sharding finalizers is called
            finalize()
            sharding_mock().finalize.assert_called_once()
            writer.finalize.assert_called_once()

            # check the output directory
            assert set(os.listdir(tmp_path)) == {
                datasets.config.DATASET_STATE_JSON_FILENAME,
                datasets.config.DATASET_INFO_FILENAME,
            }

    @pytest.mark.parametrize("path_exists", [True, False])
    def test_write_dataset(self, path_exists, tmp_path):
        tmp_path = tmp_path if path_exists else os.path.join(tmp_path, "data")

        ds = Dataset.from_dict({"obj": [0]})
        ds = ds.to_iterable_dataset(1)
        ds._format_kwargs = {"key": 0}

        class MockDatasetWriter(BaseDatasetWriter):
            SHARD_FILE_EXTENSION = "mock"
            initialize = MagicMock()
            write_batch_py = MagicMock()
            finalize = MagicMock()
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        with (
            patch("crane.core.writer.BaseDatasetWriter._consume_dataset") as consume_mock,
            patch("crane.core.writer.BaseDatasetWriter._finalize_split") as finalize_mock,
        ):
            writer = MockDatasetWriter(save_dir=tmp_path, overwrite=path_exists)
            writer.write(ds)

            # the shards, then the metadata that needs all of them to exist
            consume_mock.assert_called_once_with(ANY, tmp_path, callbacks=[])
            finalize_mock.assert_called_once_with(ANY, tmp_path)

    @pytest.mark.parametrize("path_exists", [True, False])
    def test_write_dataset_dict(self, path_exists, tmp_path):
        tmp_path = tmp_path if path_exists else os.path.join(tmp_path, "data")

        ds = Dataset.from_dict({"obj": [0]})
        ds = ds.to_iterable_dataset(1)
        ds = IterableDatasetDict({"train": ds, "test": ds})

        class MockDatasetWriter(BaseDatasetWriter):
            SHARD_FILE_EXTENSION = "mock"
            initialize = MagicMock()
            write_batch_py = MagicMock()
            finalize = MagicMock()
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        with (
            patch("crane.core.writer.BaseDatasetWriter._consume_dataset") as consume_mock,
            patch("crane.core.writer.BaseDatasetWriter._finalize_split") as finalize_mock,
        ):
            writer = MockDatasetWriter(save_dir=tmp_path, overwrite=path_exists)
            writer.write(ds)

            # every split is written, and every split is finalized
            assert {c.args[1] for c in consume_mock.mock_calls if c.args} == {
                os.path.join(tmp_path, key) for key in ds
            }
            assert {c.args[1] for c in finalize_mock.mock_calls if c.args} == {
                os.path.join(tmp_path, key) for key in ds
            }

        # check if dataset dict json exists in output directory
        assert datasets.config.DATASETDICT_JSON_FILENAME in os.listdir(tmp_path)

    def test_write_state_lists_sorted_files_only(self, tmp_path):
        ds = Dataset.from_dict({"obj": [0]}).to_iterable_dataset(1)

        class MockDatasetWriter(BaseDatasetWriter):
            SHARD_FILE_EXTENSION = "mock"
            write_batch_py = MagicMock()
            initialize = MagicMock()
            finalize = MagicMock()
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        writer = MockDatasetWriter(save_dir=tmp_path, overwrite=True, num_proc=1)

        # written in reverse so a filesystem that reports creation order is caught as well
        shards = [f"shard-{i:05d}.arrow" for i in range(20)]
        for shard in reversed(shards):
            (tmp_path / shard).touch()
        (tmp_path / "working-dir").mkdir()

        with chdir(tmp_path):
            writer._write_state(ds)

            with open(datasets.config.DATASET_STATE_JSON_FILENAME, encoding="utf-8") as f:
                state = json.load(f)

        assert [entry["filename"] for entry in state["_data_files"]] == shards


@dataclass(frozen=True)
class _PrefixedShardName(ShardName):
    """A naming policy of the shape a launcher supplies, standing in for a real one."""

    prefix: str

    def __call__(self, shard_id: int, ext: str) -> str:
        return self.prefix + super().__call__(shard_id, ext)


class TestShardNaming:
    @pytest.fixture(autouse=True)
    def in_a_worker(self):
        # opening a shard starts its buffer, which lives in the worker's context
        reset_worker_info()
        set_worker_info(rank=0, num_workers=1, seed=None)
        yield
        reset_worker_info()

    @pytest.fixture
    def writer_type(self):
        class MockDatasetWriter(BaseDatasetWriter):
            SHARD_FILE_EXTENSION = "mock"
            write_batch_py = MagicMock()
            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        return MockDatasetWriter

    def test_the_default_name_is_the_one_datasets_on_disk_already_use(self, writer_type, tmp_path):
        writer = writer_type(save_dir=str(tmp_path))
        assert writer._shard_path(42) == "shard-00042.mock"

    def test_a_policy_replaces_the_name_entirely(self, writer_type, tmp_path):
        writer = writer_type(save_dir=str(tmp_path))

        writer._set_shard_name(_PrefixedShardName("job-7-"))

        assert writer._shard_path(42) == "job-7-shard-00042.mock"

    def test_the_writer_is_handed_a_path_not_an_id(self, writer_type, tmp_path):
        writer = writer_type(save_dir=str(tmp_path))
        info = MagicMock()

        writer._open_shard(3, info=info)

        writer.initialize_shard.assert_called_once_with("shard-00003.mock", info)

    def test_the_policy_survives_being_sent_to_a_worker(self, writer_type, tmp_path):
        # the sharding controller carries the writer into every worker process, so a policy
        # that could not be pickled would name shards correctly only in the main process
        writer = writer_type(save_dir=str(tmp_path))
        writer._set_shard_name(_PrefixedShardName("job-7-"))

        revived = dill.loads(dill.dumps(writer._shard_name))

        assert revived(42, "mock") == "job-7-shard-00042.mock"


class TestShardBufferOwnership:
    """Where the batch buffer lives, which is not a detail.

    A worker receives one copy of the writer for writing and another for finalizing - they
    are pickled separately, the write path through the runner's setup and the finalize path
    through its worker callbacks. A buffer held on the writer is therefore filled by one copy
    and flushed by the other, which is to say never: every row still held when a shard closes
    is dropped, silently, and a dataset smaller than one batch comes out empty.
    """

    @pytest.fixture
    def writer_type(self):
        class MockDatasetWriter(BaseDatasetWriter):
            SHARD_FILE_EXTENSION = "mock"
            written = []

            def write_batch_arrow(self, batch):
                type(self).written.append(batch)
                return 1

            initialize_shard = MagicMock()
            finalize_shard = MagicMock()

        MockDatasetWriter.written = []
        return MockDatasetWriter

    @pytest.fixture(autouse=True)
    def in_a_worker(self):
        reset_worker_info()
        set_worker_info(rank=0, num_workers=1, seed=None)
        yield
        reset_worker_info()

    def test_a_separate_copy_can_flush_what_another_buffered(self, writer_type, tmp_path):
        writer = writer_type(save_dir=str(tmp_path), write_batch_size=1000)
        # exactly what a worker is handed: two copies that never see each other again
        finalizing_copy = deepcopy(writer)

        writer._open_shard(0, info=MagicMock())
        writer._write_batches(writer.write_batch_arrow, pa.table({"x": list(range(10))}))
        assert writer_type.written == [], "nothing should be written before a batch is full"

        finalizing_copy._close_shard(info=MagicMock())

        assert [t.num_rows for t in writer_type.written] == [10]

    def test_the_buffer_is_not_kept_on_the_writer(self, writer_type, tmp_path):
        writer = writer_type(save_dir=str(tmp_path), write_batch_size=1000)
        writer._open_shard(0, info=MagicMock())

        assert not hasattr(writer, "_buffer")
        assert get_worker_info().ctx.buffer is not None
