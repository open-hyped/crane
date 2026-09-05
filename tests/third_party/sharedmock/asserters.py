from pprint import pformat


def assert_calls_equal(expected, actual):
    """
    Check whether the given mock object (or mock method) calls are equal and
    return a nicely formatted message.
    """
    if not expected == actual:
        raise_calls_differ_error(expected, actual)


def raise_calls_differ_error(expected, actual):
    """
    Raise an AssertionError with pretty print format for the given expected
    and actual mock calls in order to ensure consistent print style for better
    readability.
    """
    expected_str = pformat(expected)
    actual_str = pformat(actual)
    msg = "\nMock calls differ!\nExpected calls:\n{}\nActual calls:\n{}".format(
        expected_str, actual_str
    )
    raise AssertionError(msg)


def assert_calls_equal_unsorted(expected, actual):
    """
    Raises an AssertionError if the two iterables do not contain the same items.

    The order of the items is ignored, but nothing else is: a call that appears more often
    than expected fails just as a missing one does.

    Both directions matter. This used to check only that every expected call was present,
    which meant a run that processed the same data twice passed - every test asserted that
    all of the data had been processed and none that nothing else had. That is how a bug
    that silently duplicated rows in the output survived the suite.
    """
    expected, actual = list(expected), list(actual)

    # `mock.call` holds the call's arguments, which are routinely dicts, so the calls are
    # not hashable and cannot be counted with a `Counter`. The suite compares a few hundred
    # at most, so counting them the slow way costs nothing worth saving.
    if len(expected) != len(actual) or any(
        expected.count(item) != actual.count(item) for item in expected
    ):
        raise_calls_differ_error(expected, actual)
