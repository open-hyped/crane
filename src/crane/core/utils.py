"""Utility module for enhanced iterator control.

This module provides common iterator functionality to simplify the processing
of iterables and queues, offering more flexible control over iteration.
"""
import math
import operator
import os
import sys
from contextlib import contextmanager
from functools import reduce
from itertools import islice
from queue import Empty, Queue
from time import perf_counter
from typing import Any, Callable, Generator, Generic, Iterable, Iterator, Tuple, TypeAlias, TypeVar

clock = perf_counter
"""A clock function used to retrieve the current time.

Returns:
    float: The current time in fractional seconds since an arbitrary point
    (typically the program's start or some system-defined moment).
"""


Sample: TypeAlias = dict[str, Any]
"""A sample of the dataset.

Represents a single data point or record in the dataset structured as a dictionary.
The keys are strings representing feature names, and the values can be of any type
(e.g., integers, floats, strings, or more complex data structures).
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


def is_python_version_less_than(major, minor: int = 0, micro: int = 0):
    """Check if the current Python version is less than the provided version.

    Args:
        major (int): Major version to compare.
        minor (int): Minor version to compare. Defaults to 0.
        micro (int): Micro version to compare. Defaults to 0.

    Returns:
        bool: True if the current Python version is less than the provided version.
    """
    current_version = sys.version_info
    target_version = (major, minor, micro)
    return current_version < target_version


if not is_python_version_less_than(3, 12):
    from itertools import batched

else:
    T = TypeVar("T")

    def batched(iterable: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
        """Yields successive n-sized batches from the given iterable.

        Args:
            iterable (Iterable[T]): An iterable to be batched.
            n (int): The size of each batch. Must be at least one.

        Raises:
            ValueError: If n is less than 1.

        Yields:
            Iterator[Tuple[T, ...]]: A generator that yields tuples containing
            n elements each from the iterable. The last batch may contain fewer
            than n elements if the total number of elements is not a multiple of n.
        """
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch


def ith_entries(iterable: Iterable[Tuple], i: int) -> Iterator:
    """Returns an iterator that yields the i-th entry from each tuple of an iterable.

    Args:
        iterable (Iterable[Tuple]): An iterable that yields tuples.
        i (int): The index of the entry to extract from each tuple.

    Returns:
        Iterator: An iterator yielding the i-th entry from each tuple.
    """
    return map(operator.itemgetter(i), iterable)


class QueueIterator:
    """An iterator for consuming items from a queue.

    This iterator retrieves items from a queue with an optional timeout and can
    terminate iteration upon encountering a sentinel value.
    """

    def __init__(self, queue: Queue, sentinel: Any = None, timeout: None | float = None) -> None:
        """Initialize the iterator.

        Args:
            queue (Queue): The queue to iterate through.
            sentinel (Any): A special value to signal the end of the queue.
            timeout (float): Timeout for waiting on queue items.
        """
        self.queue = queue
        self.sentinel = sentinel
        self.timeout = timeout

    def __iter__(self) -> Iterable[Any]:
        """Return the iterator object itself.

        Returns:
            Iterable[Any]: The iterator object itself.
        """
        return self

    def __next__(self) -> Any:
        """Retrieve the next item from the queue.

        Returns:
            Any: The next item from the queue.

        Raises:
            StopIteration: If the sentinel value is encountered or if the queue is empty.
        """
        try:
            # Get the next item from the queue with a timeout
            item = self.queue.get(timeout=self.timeout)

            # If the sentinel value is encountered, raise StopIteration to end iteration
            if item == self.sentinel:
                raise StopIteration

            return item
        except Empty:
            raise StopIteration  # noqa: B904


T = TypeVar("T")


class StoppableIterator(Generic[T]):
    """An iterator that can be stopped manually.

    This iterator allows manual interruption of iteration by calling the :func:`stop`
    method, which causes the iterator to stop yielding items.
    """

    def __init__(self, iterable: Iterable[T]) -> None:
        """Initialize the stoppable iterator.

        Args:
            iterable (Iterable[T]): The iterable to iterate over.
        """
        self.iterable = iter(iterable)
        self.stopped = False

    def stop(self) -> None:
        """Stop the iteration.

        Sets the stopped flag to True, causing the iterator to stop yielding items.
        """
        self.stopped = True

    def __iter__(self) -> Iterable[T]:
        """Return the iterator object itself.

        Returns:
            Iterable[T]: The iterator object itself.
        """
        return self

    def __next__(self) -> T:
        """Retrieve the next item from the iterable.

        Returns:
            T: The next item from the iterable.

        Raises:
            StopIteration: If iteration is stopped or if the iterable is exhausted.
        """
        if self.stopped:
            raise StopIteration

        return next(self.iterable)


T = TypeVar("T")


class TimedIterator(Generic[T]):
    """An iterator wrapper that tracks the total and smoothed average time per iteration.

    An iterator wrapper that tracks the time taken to iterate over elements
    of an iterable. It provides the total time and the smoothed average time
    per iteration using an Exponential Moving Average (EMA).
    """

    def __init__(self, iterable: Iterable[T], smoothing: float = 0.3) -> None:
        """Initializes the :class:`TimedIterator`.

        Args:
            iterable (Iterable[T]): The iterable object to track.
            smoothing (float, optional): The smoothing factor used for the EMA calculation.
                Defaults to 0.3.
        """
        self.iterable = iterable
        self.ema = EMA(smoothing=smoothing)
        self.total = 0

    def smooth_time(self) -> float:
        """Returns the smoothed average time per iteration.

        Returns:
            float: The average time per iteration based on the EMA.
        """
        return self.ema.value

    def total_time(self) -> float:
        """Returns the total accumulated time spent iterating.

        Returns:
            float: The total time spent iterating over the iterable.
        """
        return self.total

    def __iter__(self) -> Iterable[T]:
        """Returns an iterator object.

        Returns:
            Iterable[T]: The TimedIterator itself.
        """
        return self

    def __next__(self) -> T:
        """Returns the next element from the iterable and updates the time statistics.

        Tracks the time taken to retrieve the next item, updates the EMA and total time,
        and returns the next item.

        Returns:
            T: The next item from the iterable.

        Raises:
            StopIteration: When the iterable is exhausted.
        """
        # track time required for next item
        st = clock()
        item = next(self.iterable)
        # update average and return the item
        dt = clock() - st
        self.ema(dt)
        self.total += dt
        return item


T = TypeVar("T")
U = TypeVar("U")


class BatchBuffer(Generic[T, U]):
    """A buffer that collects items until a specified batch.

    It applies a function when the target batch size is reached.
    """

    def __init__(self, batch_size: int, apply_function: Callable[[list[T]], U]) -> None:
        """Initialize the :class:`BatchBuffer`.

        Args:
            batch_size (int): The number of items to collect before applying the function.
            apply_function (Callable[[List[T]], U]): The function to apply to the collected items.
        """
        self._batch_size = batch_size
        self._apply_function = apply_function
        self._buffer = []

    def add(self, item: T) -> U | None:
        """Add an item to the buffer and apply the function if the batch size is reached.

        Args:
            item (T): The item to add to the buffer.

        Returns:
            U | None: The result of applying the function if the batch size is reached; otherwise,
            returns :code:`None`.
        """
        self._buffer.append(item)
        if self.is_full():
            return self.flush()

    def flush(self) -> U | None:
        """Apply the function to the collected items and clear the buffer.

        Returns:
            U | None: The result of applying the function to the collected items.
        """
        if len(self._buffer) > 0:
            out = self._apply_function(self._buffer)
            self.clear()
            return out

    def is_full(self) -> bool:
        """Check if the buffer is full.

        Returns:
            bool: True if the buffer has reached the batch size; otherwise, False.
        """
        return len(self._buffer) >= self._batch_size

    def clear(self) -> None:
        """Clear the buffer without applying the function."""
        self._buffer.clear()


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
