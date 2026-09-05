import pickle
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from crane.core.sharding import ShardingController, ShardingStrategy, _parse_size_str
from crane.core.utils import FormatType


class TestParseSize:
    def test_parse_size_str_valid_units(self):
        assert _parse_size_str("5GB") == 5 * 1024**3
        assert _parse_size_str("200MB") == 200 * 1024**2
        assert _parse_size_str("1.5KB") == int(1.5 * 1024)
        assert _parse_size_str("2TB") == 2 * 1024**4
        assert _parse_size_str("300B") == 300

    def test_parse_size_str_with_spaces(self):
        assert _parse_size_str(" 5 GB ") == 5 * 1024**3
        assert _parse_size_str(" 200 MB") == 200 * 1024**2
        assert _parse_size_str("1.5 KB") == int(1.5 * 1024)

    def test_parse_size_str_invalid_units(self):
        with pytest.raises(ValueError):
            _parse_size_str("5XYZ")
        with pytest.raises(ValueError):
            _parse_size_str("NotASize")

    def test_parse_size_str_invalid_value(self):
        with pytest.raises(ValueError):
            _parse_size_str("abcMB")
        with pytest.raises(ValueError):
            _parse_size_str("5.5.5GB")

    def test_parse_size_str_edge_cases(self):
        assert _parse_size_str("0KB") == 0
        assert _parse_size_str("0B") == 0

        with pytest.raises(ValueError):
            _parse_size_str("")

        with pytest.raises(ValueError):
            _parse_size_str("   ")


class TestShardingController:
    def test_initialization_valid_parameters(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        controller = ShardingController(
            is_multi_processed=True,
            sharding_strategy=ShardingStrategy.FILE_SIZE,
            max_shard_size=1024,
            sample_size_key=None,
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
            formatting=FormatType.PYTHON,
        )

        assert controller._is_multi_processed is True
        assert controller._sharding_strategy == ShardingStrategy.FILE_SIZE
        assert controller._max_shard_size == 1024
        assert controller._sample_size_key is None

    def test_initialization_invalid_sample_size_key(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        with pytest.raises(ValueError):
            ShardingController(
                is_multi_processed=True,
                sharding_strategy=ShardingStrategy.SAMPLE_ITEM,
                max_shard_size=1024,
                sample_size_key=None,
                initialize_shard=initialize_shard,
                finalize_shard=finalize_shard,
                formatting=FormatType.PYTHON,
            )

    def test_initialization_invalid_max_shard_size_for_non_file_size(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        with pytest.raises(ValueError):
            ShardingController(
                is_multi_processed=True,
                sharding_strategy=ShardingStrategy.SAMPLE_ITEM,
                max_shard_size="5GB",
                sample_size_key="key",
                initialize_shard=initialize_shard,
                finalize_shard=finalize_shard,
                formatting=FormatType.PYTHON,
            )

    def test_warning_on_sample_size_key_for_non_sample_item(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        with pytest.warns(UserWarning):
            ShardingController(
                is_multi_processed=True,
                sharding_strategy=ShardingStrategy.FILE_SIZE,
                max_shard_size=1024,
                sample_size_key="key",
                initialize_shard=initialize_shard,
                finalize_shard=finalize_shard,
                formatting=FormatType.PYTHON,
            )

    @pytest.mark.parametrize("is_multi_processed", [True, False])
    def test_initialize_and_finalize(self, is_multi_processed):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        controller = ShardingController(
            is_multi_processed=is_multi_processed,
            sharding_strategy=ShardingStrategy.FILE_SIZE,
            max_shard_size=1024,
            sample_size_key=None,
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
            formatting=FormatType.PYTHON,
        )

        for i in range(3):
            initialize_shard.reset_mock()
            finalize_shard.reset_mock()

            controller.initialize()
            initialize_shard.assert_called_once_with(i)

            controller.finalize()
            finalize_shard.assert_called_once()

        # cannot initialize new shard without finalize in between
        controller.initialize()
        with pytest.raises(AssertionError):
            controller.initialize()
        # finalize
        controller.finalize()

        # shard finalizer is not called when no shard is initialized
        finalize_shard.reset_mock()
        controller.finalize()
        assert not finalize_shard.called

    def test_sample_count_strategy(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        max_shard_size = 8
        controller = ShardingController(
            is_multi_processed=False,
            sharding_strategy=ShardingStrategy.SAMPLE_COUNT,
            max_shard_size=max_shard_size,
            sample_size_key=None,
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
            formatting=FormatType.PYTHON,
        )
        # initialize the controller
        controller.initialize()
        # reset the mocks
        initialize_shard.reset_mock()
        finalize_shard.reset_mock()

        for _ in range(max_shard_size // 2):
            controller.callback({"key": [42] * 2})
            controller.update(42)
            # no new shard needed to be generated yet
            assert not finalize_shard.called
            assert not initialize_shard.called

        # this update should kick of a new shard
        controller.callback({"key": [42] * 2})
        controller.update(42)

        finalize_shard.assert_called_once()
        initialize_shard.assert_called_once()

    def test_sample_item_strategy(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        max_shard_size = 153
        controller = ShardingController(
            is_multi_processed=False,
            sharding_strategy=ShardingStrategy.SAMPLE_ITEM,
            max_shard_size=max_shard_size,
            sample_size_key="key",
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
            formatting=FormatType.ARROW,
        )
        # initialize the controller
        controller.initialize()
        # reset the mocks
        initialize_shard.reset_mock()
        finalize_shard.reset_mock()

        batch = pa.table({"key": [42]})

        for _ in range(max_shard_size // 42 + 1):
            controller.callback(batch)
            controller.update(1)
            # no new shard needed to be generated yet
            assert not finalize_shard.called
            assert not initialize_shard.called

        # this update should kick of a new shard
        controller.callback(batch)
        controller.update(1)

        finalize_shard.assert_called_once()
        initialize_shard.assert_called_once()

    def test_file_size_strategy(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        max_shard_size = 153
        controller = ShardingController(
            is_multi_processed=False,
            sharding_strategy=ShardingStrategy.FILE_SIZE,
            max_shard_size=max_shard_size,
            sample_size_key=None,
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
            formatting=FormatType.PYTHON,
        )
        # initialize the controller
        controller.initialize()
        # reset the mocks
        initialize_shard.reset_mock()
        finalize_shard.reset_mock()

        for _ in range(max_shard_size // 42 + 1):
            controller.callback({"key": [0]})
            controller.update(42)
            # no new shard needed to be generated yet
            assert not finalize_shard.called
            assert not initialize_shard.called

        # this update should kick of a new shard
        controller.callback({"key": [0]})
        controller.update(42)

        finalize_shard.assert_called_once()
        initialize_shard.assert_called_once()

    def test_no_shard_is_opened_without_a_batch(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        controller = ShardingController(
            is_multi_processed=False,
            sharding_strategy=ShardingStrategy.FILE_SIZE,
            max_shard_size=1024,
            sample_size_key=None,
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
            formatting=FormatType.PYTHON,
        )

        # the whole life of a worker the data never reached: it is started and finished
        # without a single batch ever arriving
        controller.finalize()

        # so it must leave nothing behind - an opened shard here is an empty file in the
        # written dataset, and one a parquet `_metadata` cannot even name. What used to
        # open it is the writer asking for a shard when a worker starts, which
        # `test_write_split` in test_writer.py is what pins down.
        assert not initialize_shard.called
        assert not finalize_shard.called

    def test_first_batch_opens_the_shard(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        controller = ShardingController(
            is_multi_processed=False,
            sharding_strategy=ShardingStrategy.SAMPLE_COUNT,
            max_shard_size=1024,
            sample_size_key=None,
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
            formatting=FormatType.PYTHON,
        )

        controller.callback({"key": [42]})
        controller.update(42)

        initialize_shard.assert_called_once_with(0)
        assert not finalize_shard.called

        # the shard is nowhere near full, so the next batch goes into the same one
        controller.callback({"key": [42]})
        controller.update(42)

        initialize_shard.assert_called_once()

    def test_none_strategy_opens_one_shard_and_never_rolls_over(self):
        initialize_shard = MagicMock()
        finalize_shard = MagicMock()

        controller = ShardingController(
            is_multi_processed=False,
            sharding_strategy=ShardingStrategy.NONE,
            max_shard_size=None,
            sample_size_key=None,
            initialize_shard=initialize_shard,
            finalize_shard=finalize_shard,
            formatting=FormatType.PYTHON,
        )

        for _ in range(10):
            controller.callback({"key": [42] * 100})
            controller.update(4096)

        # sharding is off, so everything goes into the one shard the first batch opened
        initialize_shard.assert_called_once_with(0)
        assert not finalize_shard.called

        controller.finalize()
        finalize_shard.assert_called_once()

    def test_a_pickled_copy_shares_the_open_shard(self):
        # A worker receives the controller twice - once with the process object and once
        # pickled into the worker setup - so the copy that opens a shard is not the copy
        # that is asked to close it. Both must mean the same shard.
        _SHARD_CALLS.clear()

        controller = ShardingController(
            is_multi_processed=False,
            sharding_strategy=ShardingStrategy.SAMPLE_COUNT,
            max_shard_size=4,
            sample_size_key=None,
            initialize_shard=_record_initialize_shard,
            finalize_shard=_record_finalize_shard,
            formatting=FormatType.PYTHON,
        )
        copy = pickle.loads(pickle.dumps(controller))

        # the copy inside the write function opens the shard
        copy.callback({"key": [42]})
        copy.update(42)
        assert _SHARD_CALLS == [("initialize", 0)]

        # the copy the worker was started with sees that same shard as open: it neither
        # opens a second one for the next batch ...
        controller.callback({"key": [42]})
        controller.update(42)
        assert _SHARD_CALLS == [("initialize", 0)]

        # ... nor mistakes it for "nothing open" when the worker finishes, which would
        # leave the shard the other copy opened unfinalized
        controller.finalize()
        assert _SHARD_CALLS == [("initialize", 0), ("finalize", None)]


_SHARD_CALLS: list[tuple[str, None | int]] = []
"""The shard hooks that were called, in order. Written by the two functions below.

A module-level recorder rather than a mock: the hooks have to survive being pickled with
the controller, and both copies must reach the same recorder.
"""


def _record_initialize_shard(shard_id: int) -> None:
    """Record a shard being opened.

    Args:
        shard_id (int): The shard being opened.
    """
    _SHARD_CALLS.append(("initialize", shard_id))


def _record_finalize_shard() -> None:
    """Record the open shard being closed."""
    _SHARD_CALLS.append(("finalize", None))


def _noop_shard(shard_id: int) -> None:
    """Picklable stand-in for the shard initialization and finalization hooks.

    Args:
        shard_id (int): The shard the hook is called for.
    """


class TestShardingControllerPickling:
    """The controller is sent to the workers by pickle under the `spawn` start method."""

    @pytest.mark.parametrize("formatting", list(FormatType))
    @pytest.mark.parametrize("strategy", list(ShardingStrategy))
    def test_controller_is_picklable(self, strategy, formatting):
        controller = ShardingController(
            is_multi_processed=False,
            sharding_strategy=strategy,
            max_shard_size=1024,
            sample_size_key=("size" if strategy is ShardingStrategy.SAMPLE_ITEM else None),
            initialize_shard=_noop_shard,
            finalize_shard=_noop_shard,
            formatting=formatting,
        )

        reconstructed = pickle.loads(pickle.dumps(controller))
        assert reconstructed._batch_size_fn is controller._batch_size_fn
