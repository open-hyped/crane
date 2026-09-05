"""Tests for the shared-mock call assertions the runner tests rely on.

These guard the assertion itself rather than any crane behaviour. The multiprocessing
tests state their expectations entirely through :func:`assert_calls_equal_unsorted`, so a
weakness here is invisible everywhere and silently weakens every test that uses it - which
is what happened: it once checked only that each expected call was present, and a bug that
made the runner write every row of a shard twice passed the whole suite.
"""

from unittest.mock import call

import pytest

from tests.third_party.sharedmock.asserters import assert_calls_equal_unsorted


class TestAssertCallsEqualUnsorted:
    def test_accepts_the_same_calls(self):
        assert_calls_equal_unsorted([call(1), call(2)], [call(1), call(2)])

    def test_ignores_order(self):
        # which worker picks up which part of the data is decided at runtime
        assert_calls_equal_unsorted([call(1), call(2)], [call(2), call(1)])

    def test_rejects_a_missing_call(self):
        with pytest.raises(AssertionError):
            assert_calls_equal_unsorted([call(1), call(2)], [call(1)])

    def test_rejects_an_extra_call(self):
        # the blind spot: work that was never asked for used to pass
        with pytest.raises(AssertionError):
            assert_calls_equal_unsorted([call(1)], [call(1), call(2)])

    def test_rejects_a_repeated_call(self):
        # the exact shape of a duplicated row reaching the writer twice
        with pytest.raises(AssertionError):
            assert_calls_equal_unsorted([call(1), call(2)], [call(1), call(2), call(2)])

    def test_rejects_unhashable_arguments_being_repeated(self):
        # samples are dicts, so the calls cannot be counted with a `Counter`
        with pytest.raises(AssertionError):
            assert_calls_equal_unsorted([call({"a": 1})], [call({"a": 1}), call({"a": 1})])
