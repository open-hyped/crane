"""Utility module for enhanced iterator control.

This module provides common iterator functionality to simplify the processing
of iterables and queues, offering more flexible control over iteration.
"""
import math
import os
from contextlib import contextmanager
from functools import reduce
from time import perf_counter
from typing import Any, Callable, Generator

clock = perf_counter
"""A clock function used to retrieve the current time.

Returns:
    float: The current time in fractional seconds since an arbitrary point
    (typically the program's start or some system-defined moment).
"""


@contextmanager
def chdir(new_dir: str) -> Generator[None, None, None]:
    """Context manager for temporarily changing the working directory.

    Args:
        new_dir (str): The directory to change to temporarily.

    Yields:
        None
    """
    original_dir = os.getcwd()  # Save the current working directory
    os.chdir(new_dir)  # Change to the new directory
    try:
        yield  # Yield control back to the caller
    finally:
        os.chdir(original_dir)  # Restore the original directory


class Compose(object):
    """Composes an arbitrary number of functions into a single function.

    The composed function applies the input functions from right to left (i.e.,
    the last function in the list is applied first, and the first function is applied last).

    Example:
        If :code:`Compose(f, g, h)` is called with input :code:`x`, it returns :code:`f(g(h(x)))`.

    Args:
        *functions (Callable[[Any], Any]): An arbitrary number of functions to compose.
            Each function must accept the output of the subsequent function (or the initial input).
            Must contain at least one function.

    Returns:
        Callable[[Any], Any]: A function that applies the composed functions sequentially from
        right to left.
    """

    def __init__(self, *functions: Callable[[Any], Any]) -> None:
        """Initialize the :class:`Compose` object with the provided functions.

        Args:
            *functions (Callable[[Any], Any]): Functions to be composed.
        """
        assert len(functions) > 0, "At least one function must be provided"
        self._functions = tuple(reversed(functions))

    def __call__(self, x: Any) -> Any:
        """Applies the composed functions to the input.

        Args:
            x (Any): The initial input to be passed through the composed functions.

        Returns:
            Any: The result of applying the composed functions sequentially.
        """
        return reduce(lambda x, f: f(x), self._functions, x)


class RunAll(object):
    """Runs an arbitrary number of functions sequentially with the same arguments.

    Each function in the list is called with the provided arguments, and their execution
    order is from first to last. No function's output is used as input for the next.

    Example:
        If :code:`RunAll(f, g, h)` is called with arguments :code:`x, y`, it executes:

        .. code-block:: python

            f(x, y)
            g(x, y)
            h(x, y)

    Args:
        *functions (Callable[[Any], Any]): An arbitrary number of functions to be run
            sequentially. Each function must accept the same arguments.

    Returns:
        None
    """

    def __init__(self, *functions: Callable[[Any], Any]) -> None:
        """Initialize the :class:`RunAll` object with the provided functions.

        Args:
            *functions (Callable[[Any], Any]): Functions to be executed sequentially.
        """
        self._functions = functions

    def __call__(self, *args, **kwargs) -> None:
        """Executes all functions with the provided arguments and keyword arguments.

        Args:
            *args (Any): Positional arguments to be passed to each function.
            **kwargs (Any): Keyword arguments to be passed to each function.

        Returns:
            None
        """
        for f in self._functions:
            f(*args, **kwargs)


class EMA(object):
    r"""Calculate the exponential moving average (EMA).

    The EMA assigns progressively lower weights to older values, giving more
    significance to recent measurements. It smooths out fluctuations by
    applying a smoothing factor to the most recent value, combining it with
    the previous EMA result.

    The weight of past values is controlled by the smoothing factor. A higher
    smoothing factor gives more influence to recent values and less to older ones.

    The smoothing factor, \( \alpha \), is in the range [0, 1], where:
    - \( \alpha = 0 \) retains the old value entirely.
    - \( \alpha = 1 \) adopts the new value entirely.
    """

    def __init__(self, smoothing: float = 0.3) -> None:
        """Initialize the EMA.

        Args:
            smoothing (float): Smoothing factor controlling the
                weight of recent values. Default is 0.3.
        """
        self.alpha = smoothing
        self.last_ema = None
        self.calls = 0

    @property
    def value(self) -> float:
        """Return the current EMA value.

        Returns:
            float: The current EMA value or 0.0 if no values have been recorded.
        """
        # apply the initialization bias normalization
        beta = 1 - self.alpha
        return 0 if self.last_ema is None else (self.last_ema / (1 - beta**self.calls))

    def __call__(self, value: float) -> float:
        """Update the EMA with a new value and return the updated value.

        Args:
            value (float): New value to include in the EMA.

        Returns:
            float: The updated EMA value after incorporating the new value.
        """
        self.update(value)
        return self.value

    def update(self, value: float) -> float:
        """Update the EMA with a new value.

        This method incorporates the new value with the previous EMA using
        the smoothing factor.

        Args:
            value (float): The new measured value to update the EMA.

        Returns:
            float: The updated EMA value.
        """
        beta = 1 - self.alpha
        # update the ema
        self.last_ema = (
            value if self.last_ema is None else self.alpha * value + beta * self.last_ema
        )
        self.calls += 1


class TimeWeightedEMA(object):
    r"""Calculate the time-weighted exponential moving average (EMA).

    The time-weighted EMA assigns greater relevance to more recent measurements while accounting
    for the time elapsed between measurements. Older measurements are progressively less relevant
    based on their age relative to the most recent measurement.

    The weight of past measurements is controlled by the decay rate. A higher decay rate causes
    older measurements to lose influence more rapidly.

    The decay factor is calculated as:
    .. math::
        \text{decay}_t = e^{-\lambda \cdot \Delta t}

    where \( \Delta t \) is the time difference between the current timestamp and the
    previous one.

    The decay rate can be derived from a desired half-life using the formula:
    .. math::
        \lambda = \frac{\ln(2)}{\text{half-life}}

    where the half-life is the time period after which the measurement's weight is reduced
    by half.
    """

    def __init__(self, decay_rate: float) -> None:
        """Initialize the EMA with a given decay rate.

        Args:
            decay_rate (float): Decay rate controlling the weight of past measurements.
        """
        self.decay_rate = decay_rate
        self.last_ema = None
        self.previous_timestamp = None
        self.calls = 0
        self.norm_factor = 1.0

    @property
    def value(self) -> float:
        """Get the current EMA value.

        Returns:
            float: The current EMA value or 0.0 if no measurements have been recorded.
        """
        return (self.last_ema / self.norm_factor) if self.last_ema is not None else 0.0

    @property
    def timestamp(self) -> float:
        """Get the timestamp of the last update.

        Returns:
            float: The timestamp of the previous update or None if no updates have been made.
        """
        return self.previous_timestamp

    def __call__(self, timestamp: float, value: float) -> float:
        """Update the EMA with a new value and return the updated EMA.

        This method allows the object to be called like a function, updating the EMA
        and returning the new value.

        Args:
            timestamp (float): The current time when the value is measured.
            value (float): The new measured value to update the EMA.

        Returns:
            float: The updated EMA value.
        """
        self.update(timestamp, value)
        return self.value

    def update(self, timestamp: float, value: float) -> float:
        """Update the EMA with a new value, considering the time-weighted decay.

        The method calculates the decay factor based on the time elapsed between the
        current and previous measurements, then applies the decay to update the EMA.

        Args:
            timestamp (float): The current time when the value is measured.
            value (float): The new measured value to update the EMA.

        Returns:
            float: The updated EMA value.
        """
        if self.last_ema is None:
            # Initialize EMA with the first measurement
            self.last_ema = value
        else:
            if self.previous_timestamp is not None:
                # Calculate the time difference
                delta_t = timestamp - self.previous_timestamp
                # Calculate the decay factor based on the time difference
                decay = math.exp(-self.decay_rate * delta_t)
                # Update EMA considering time-weighted decay
                self.last_ema = decay * value + (1 - decay) * self.last_ema
                # Update cumulative weight
                self.norm_factor = decay + (1 - decay) * self.norm_factor

        # Update state
        self.previous_timestamp = timestamp
        self.calls += 1
