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
