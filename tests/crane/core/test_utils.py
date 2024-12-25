import importlib
import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import crane.core.utils
from crane.core.utils import EMA, Compose, RunAll, TimeWeightedEMA, chdir, ith_entries


def test_chdir() -> None:
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir1:
        orig_dir = os.getcwd()

        # Test the chdir context manager
        with chdir(temp_dir1):
            assert os.getcwd().endswith(os.path.abspath(temp_dir1))  # Should be in temp_dir2

        # After exiting the context, should be back to temp_dir1
        assert os.getcwd().endswith(os.path.abspath(orig_dir))


def test_chdir_exception() -> None:
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir1:
        orig_dir = os.getcwd()

        # Test the chdir context manager with an exception
        with pytest.raises(RuntimeError):
            with chdir(temp_dir1):
                assert os.getcwd().endswith(os.path.abspath(temp_dir1))  # Should be in temp_dir2
                raise RuntimeError("Test exception")  # Raise an exception

        # After exiting the context, should be back to temp_dir1
        assert os.getcwd().endswith(os.path.abspath(orig_dir))


@patch("crane.core.utils.is_python_version_less_than", return_value=(3, 11))
def test_batched(mock_is_python_version_less_than: MagicMock) -> None:
    # reload module to make sure to use the <3.12 implementation
    importlib.reload(crane.core.utils)
    from crane.core.utils import batched

    # check batches
    assert list(batched(range(100), 10)) == [tuple(range(i, i + 10)) for i in range(0, 100, 10)]

    # batch size to small
    with pytest.raises(ValueError):
        next(batched(range(10), 0))


def test_ith_entries() -> None:
    it = [(i, i + 1) for i in range(10)]

    assert list(ith_entries(it, 0)) == list(range(10))
    assert list(ith_entries(it, 1)) == list(range(1, 11))


def test_compose() -> None:
    f, g, x = MagicMock(), MagicMock(), MagicMock()
    assert Compose(f, g)(x) == f(g(x))


def test_run_all() -> None:
    f, g, x = MagicMock(), MagicMock(), MagicMock()
    RunAll(f, g)(x)
    # make sure functions are called
    f.assert_called_once_with(x)
    g.assert_called_once_with(x)


def test_calculate_ema():
    measurements = [100, 150, 200, 250]  # Measurements taken over time

    ema = EMA(smoothing=0.3)

    for value in measurements:
        ema(value)

    # Validate the result against expected values
    assert ema.value is not None
    assert isinstance(ema.value, float)
    # Verify the calculated EMA against an expected value
    assert math.isclose(ema.value, 228.12212133175416, rel_tol=1e-5)


def test_calculate_time_weighted_ema():
    measurements = {
        1695419000: 100,  # Timestamp: 1695419000, Measurement: 100
        1695419100: 150,  # 100 seconds later
        1695419800: 200,  # 700 seconds later
        1695420400: 250,  # 600 seconds later
    }

    ema = TimeWeightedEMA(decay_rate=0.001)

    for timestamp, value in measurements.items():
        ema(timestamp, value)

    # Validate the result against expected values
    assert ema.value is not None
    assert isinstance(ema.value, float)
    # Verify the calculated EMA
    assert math.isclose(ema.value, 215.00310219329043, rel_tol=1e-5)
