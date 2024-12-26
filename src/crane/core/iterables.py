"""Custon ExampleIterable Implementations to control and monitor data flow."""
from __future__ import annotations

from copy import copy
from functools import partial
from queue import Empty, Queue
from typing import Any, Iterable

import pyarrow as pa
from datasets.formatting import PythonFormatter
from datasets.iterable_dataset import _BaseExamplesIterable

from .utils import EMA, Sample, clock


class BaseExamplesIterable(_BaseExamplesIterable):
    """Base class for custom examples iterable implementations."""

    def __init__(self, ex_iterable: _BaseExamplesIterable) -> None:
        """Initialize the BaseExamplesIterable.

        Args:
            ex_iterable (_BaseExamplesIterable): The underlying examples iterable.
        """
        super(BaseExamplesIterable, self).__init__()
        self.ex_iterable = ex_iterable

    def __iter__(self) -> Iterable[tuple[str, Sample]]:
        """Iterate over the examples.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    @property
    def iter_arrow(self) -> Iterable[tuple[str, pa.Table]]:
        """Get the arrow iterator.

        Returns:
            Iterable[pa.Table]: Data iterator of pyarrow tables.
        """
        return self._iter_arrow

    def _iter_arrow(self) -> Iterable[tuple[str, pa.Table]]:
        """The arrow data iterator.

        Implements the Iterator of pyarrow tables.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    @property
    def num_shards(self) -> int:
        """Get the number of shards of the underlying dataset.

        Returns:
            int: The number of shards.
        """
        return self.ex_iterable.num_shards

    def _init_state_dict(self) -> dict:
        """Initialize and return the state dictionary.

        This method initializes a dictionary containing the state of the
        :class:`BaseExamplesIterable` instance, including the state of the
        underlying :code:`ex_iterable`.

        Returns:
            dict: A dictionary representing the state of the instance.
        """
        self._state_dict = {"ex_iterable": self.ex_iterable._init_state_dict()}
        return self._state_dict


class StoppableExamplesIterable(BaseExamplesIterable):
    """Examples iterable that can be stopped and resumed."""

    def __init__(self, ex_iterable: _BaseExamplesIterable) -> None:
        """Initialize the :class:`StoppableExamplesIterable`.

        Args:
            ex_iterable (_BaseExamplesIterable): The underlying examples iterable.
        """
        super(StoppableExamplesIterable, self).__init__(ex_iterable)
        self._flag = False
        self._iter: None | Iterable[tuple[str, pa.Table]] = None

    def init_iter(self) -> None:
        """Initialize the internal iterator."""
        self._iter = self.ex_iterable.iter_arrow()

    def stop(self) -> None:
        """Stop the iteration."""
        self._flag = True

    def resume(self) -> None:
        """Resume the iteration."""
        self._flag = False

    def __iter__(self) -> Iterable[tuple[str, Sample]]:
        """Iterate over the examples.

        Yields:
            tuple[key, Sample]: A tuple containing a key and the formatted example.
        """
        if self._flag:
            return

        if self._iter is None:
            self.init_iter()

        formatter = PythonFormatter()
        for key, pa_table in self._iter:
            for i, item in enumerate(pa_table.to_reader(max_chunksize=1)):
                yield f"{key}_{i}", formatter.format_row(item)
            if self._flag:
                return

    def _iter_arrow(self) -> Iterable[tuple[str, pa.Table]]:
        """The arrow data iterator.

        Implements the Iterator of pyarrow tables.

        Yields:
            tuple[str, pa.Table]: A tuple containing a key and pyarrow table.
        """
        if self._flag:
            return

        if self._iter is None:
            self.init_iter()

        try:
            while not self._flag:
                yield next(self._iter)
        except StopIteration:
            self._iter = None


class TimedExamplesIterable(BaseExamplesIterable):
    """Examples iterable that tracks and exposes processing time statistics.

    This class wraps an existing examples iterable and tracks processing time
    metrics during iteration. It calculates and exposes both the exponentially
    smoothed average (EMA) processing time per example and the total processing
    time spent iterating over the entire dataset.
    """

    def __init__(self, ex_iterable: _BaseExamplesIterable, smoothing: float = 0.3) -> None:
        """Initialize the :class:`TimedExamplesIterable`.

        Args:
            ex_iterable (_BaseExamplesIterable): The underlying examples iterable.
            smoothing (float, optional): Smoothing factor for the EMA calculation.
                Defaults to 0.3.
        """
        super(TimedExamplesIterable, self).__init__(ex_iterable)
        self.ema = EMA(smoothing=smoothing)
        self.total = 0.0

    def smooth_time(self) -> float:
        """Returns the current exponentially smoothed average (EMA) processing time per example."""
        return self.ema.value

    def total_time(self) -> float:
        """Returns the total processing time spent iterating over the entire dataset."""
        return self.total

    def __iter__(self) -> Iterable[tuple[str, Sample]]:
        """Iterate over the examples, tracking and updating processing time statistics.

        Yields:
            Tuple[str, Sample]: A tuple containing the key and the next item from the
                underlying iterable.
        """
        st = clock()
        for key, item in self.ex_iterable:
            # get new time and compute time delta
            nt = clock()
            dt = nt - st
            # update tracked metrics
            self.ema(dt)
            self.total += dt
            # yield item and update start time
            yield (key, item)
            st = nt

    def _iter_arrow(self) -> Iterable[pa.Table]:
        """Iterates over the Arrow tables of the underlying iterable, tracking processing time.

        Yields:
            Tuple[str, pa.Table]: A tuple containing the key and the next Arrow table.
                Returns None if the underlying iterable's :code:`iter_arrow` returns None.
        """
        it = self.ex_iterable.iter_arrow()
        if it is None:
            return None

        st = clock()
        for key, pa_table in it:
            # get new time and compute time delta
            nt = clock()
            dt = nt - st
            # update tracked metrics
            self.ema(dt / pa_table.num_rows)
            self.total += dt
            # yield item and update start time
            yield (key, pa_table)
            st = nt


class PreQueueExamplesIterable(BaseExamplesIterable):
    """Examples iterable that adds a key to the metadata of each Arrow table.

    This class prepares an examples iterable for use with a queue by adding
    a unique key to the metadata of each Arrow table. This key is used to
    identify the origin of the table when it's retrieved from the queue.
    """

    def __init__(self, ex_iterable: _BaseExamplesIterable, key: str = "_key") -> None:
        """Initialize the :class:`PreQueueExamplesIterable`.

        Args:
            ex_iterable (_BaseExamplesIterable): The underlying examples iterable.
            key (str, optional): The key to add to the metadata. Defaults to :code:`"_key"`.
        """
        super().__init__(ex_iterable)
        self.key = key

    def __iter__(self) -> Iterable[tuple[str, Sample]]:
        """Iterate over the examples.

        Raises:
            NotImplementedError: This method should not be called directly as it is
                intended for use with Arrow tables.
        """
        raise NotImplementedError()

    def _iter_arrow(self) -> Iterable[tuple[str, pa.Table]]:
        """Iterate over the Arrow tables, adding a key to the metadata.

        Yields:
            Tuple[str, pa.Table]: A tuple containing the key and the modified Arrow table
                with the added metadata.
        """
        for key, pa_table in self.ex_iterable._iter_arrow():
            metadata = pa_table.schema.metadata or {}
            yield key, pa_table.replace_schema_metadata(
                metadata=metadata | {self.key.encode(): key.encode()}
            )


class QueueExamplesIterable(BaseExamplesIterable):
    """Examples iterable that retrieves data from a queue.

    This class consumes Arrow tables from a queue, extracting the key from
    the metadata and yielding the tables. It's designed to work in conjunction
    with :class:`PreQueueExamplesIterable`, which adds the key to the metadata
    before putting the tables into the queue.
    """

    @classmethod
    def prepare_ex_iterable(cls, ex_iterable: _BaseExamplesIterable) -> _BaseExamplesIterable:
        """Prepare an examples iterable for use with a queue.

        This method wraps the given iterable with :class:`PreQueueExamplesIterable`
        to add keys to the metadata of the Arrow tables.

        Args:
            ex_iterable (_BaseExamplesIterable): The examples iterable to prepare.

        Returns:
            _BaseExamplesIterable: The prepared examples iterable.
        """
        return PreQueueExamplesIterable(ex_iterable, key="_key")

    def __init__(
        self,
        queue: Queue,
        sentinel: Any = None,
        timeout: None | float = None,
        num_shards: None | int = None,
    ) -> None:
        """Initialize the :class:`QueueExamplesIterable`.

        Args:
            queue (Queue): The queue to retrieve data from.
            sentinel (None | Any): The sentinel value to signal the end of the queue.
                Defaults to :code:`None`.
            timeout (None | float): Timeout for queue `get` operations, in seconds.
                If None, the :code:`get` operation will block indefinitely. Defaults to
                :code:`None`.
            num_shards (None | int): The number of shards. Defaults to :code:`None`.
        """
        super().__init__(None)
        self.queue = queue
        self.sentinel = sentinel
        self.timeout = timeout
        self.n_shards = num_shards

    def __iter__(self) -> Iterable[tuple[str, Sample]]:
        """Iterate over the examples retrieved from the queue.

        Yields:
            Tuple[str, Sample]: A tuple containing the key and the formatted example.
        """
        formatter = PythonFormatter()
        for key, pa_table in self._iter_arrow():
            for i, item in enumerate(pa_table.to_reader(max_chunksize=1)):
                yield f"{key}_{i}", formatter.format_row(item)

    def _iter_arrow(self) -> Iterable[tuple[str, pa.Table]]:
        """Iterate over the Arrow tables retrieved from the queue.

        Yields:
            Tuple[str, pa.Table]: A tuple containing the key and the Arrow table.
        """
        while True:
            try:
                getter = partial(self.queue.get, timeout=self.timeout)
                for pa_table in iter(getter, self.sentinel):
                    metadata = pa_table.schema.metadata
                    key = metadata.pop(b"_key").decode()
                    yield key, pa_table.replace_schema_metadata(metadata=metadata)
            except Empty:
                break

    @property
    def num_shards(self) -> int:
        """Get the number of shards."""
        return self.n_shards

    def _init_state_dict(self) -> dict:
        """Initialize and return the state dictionary.

        Returns:
            dict: An empty dictionary.
        """
        self._state_dict = {}
        return self._state_dict


class ExamplesIterablePipeline(list[_BaseExamplesIterable]):
    """Pipeline of huggingface's :class:`_BaseExamplesIterable` blocks.

    This class manages a sequence of processing steps applied to a dataset,
    allowing for chaining and copying of processing pipelines.
    """

    @property
    def src_iterable(self) -> None | _BaseExamplesIterable:
        """Get the source iterable for the pipeline.

        Returns:
            _BaseExamplesIterable: The source iterable for the pipeline.

        Raises:
            AssertionError: If the pipeline is empty.
        """
        assert len(self) > 0, "Pipeline is empty; no source iterable available."
        return self[0].ex_iterable

    def copy(self) -> ExamplesIterablePipeline:
        """Create a copy of the pipeline with each step copied.

        Returns:
            ExamplesIterablePipeline: A new pipeline instance with copied steps.
        """
        first = copy(self[0])
        first.ex_iterable = None

        pipeline = ExamplesIterablePipeline([first])

        for step in map(copy, self[1:]):
            step.ex_iterable = pipeline[-1]
            pipeline.append(step)

        return pipeline

    def __call__(self, ex_iterable: _BaseExamplesIterable) -> _BaseExamplesIterable:
        """Apply the pipeline to a given examples iterable.

        Args:
            ex_iterable (_BaseExamplesIterable): The examples iterable to apply
                the pipeline to.

        Returns:
            _BaseExamplesIterable: The examples iterable resulting from applying
                the pipeline.
        """
        # (shallow) copy the pipeline to avoid breaking the
        # ex_iterable references in other instances of the pipeline
        pipeline = self.copy()
        pipeline[0].ex_iterable = ex_iterable
        # return the output examples iterable
        return pipeline[-1]

    def __str__(self) -> str:
        """String representation of the pipeline."""
        return "[" + ", ".join([type(step).__name__ for step in self]) + "]"
