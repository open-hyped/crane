"""Regrouping the batches a pipeline delivers into the batches a writer was asked for.

The batches flowing through a pipeline are sized for throughput - small enough to keep
memory flat and workers fed - and rebatched along the way by every :func:`map` the caller
applied. What a writer is handed is therefore whatever survived that, not what it asked for,
and a writer that passes it straight to disk lets the plumbing decide the layout of the file.

The asymmetry is what makes this worth guarding against. Writing a batch per row costs the
writer almost nothing, so nothing pushes back where the mistake is made, while formats that
carry per-batch metadata make every reader pay for it afterwards - on a real corpus here,
written a row at a time, the difference was 125 seconds to open the dataset against 0.09.

Both batch formats a writer can receive are handled, so a writer of either kind gets the
size it asked for without knowing this happened.
"""

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import pyarrow as pa

BatchT: TypeAlias = pa.Table | dict[str, list[Any]]
"""A batch, in either of the formats a writer receives."""


def num_rows(batch: BatchT) -> int:
    """Count the rows in a batch of either format.

    Args:
        batch (BatchT): The batch to measure.

    Returns:
        int: The number of rows.
    """
    if isinstance(batch, pa.Table):
        return batch.num_rows

    return len(next(iter(batch.values()), []))


def concat(batches: list[BatchT]) -> BatchT:
    """Join batches of either format into one.

    Args:
        batches (list[BatchT]): The batches to join, all of the same format and columns.

    Returns:
        BatchT: The batches as one.
    """
    if isinstance(batches[0], pa.Table):
        # `combine_chunks` is what makes the result one batch rather than one per table
        # joined: a writer emits per chunk, so concatenating alone would keep them separate
        # and the regrouping would have achieved nothing.
        return pa.concat_tables(batches).combine_chunks()

    return {key: [row for batch in batches for row in batch[key]] for key in batches[0]}


def slice_batch(batch: BatchT, offset: int, length: None | int = None) -> BatchT:
    """Take a range of rows out of a batch of either format.

    Args:
        batch (BatchT): The batch to slice.
        offset (int): The first row to take.
        length (None | int): How many rows to take, or None for all that remain.

    Returns:
        BatchT: The requested rows.
    """
    if isinstance(batch, pa.Table):
        return batch.slice(offset) if length is None else batch.slice(offset, length)

    end = None if length is None else offset + length
    return {key: values[offset:end] for key, values in batch.items()}


@dataclass
class BatchBuffer(object):
    """Regroups batches into a fixed size, whatever size they arrive in.

    Emits batches of exactly :attr:`size` and keeps the remainder, rather than handing back
    whatever has piled up. Flushing the pile would leave the result depending on the arrival
    sizes after all - batches of 256 gathered to a threshold of 500 give 512, batches of 8
    give 504 - which is the coupling this exists to break.

    One buffer belongs to one open shard: :func:`take` is what a shard's finalization calls,
    and a shard closed without it loses every row still waiting.
    """

    size: int
    """Rows in each batch handed back by :func:`add`."""

    _pending: list[BatchT] = field(default_factory=list)
    _rows: int = 0

    def add(self, batch: BatchT) -> list[BatchT]:
        """Take a batch, and hand back any full batches it completes.

        Returns a list because one arrival can complete several: a caller handing over more
        rows than :attr:`size` should not have to call again to get them all out.

        Args:
            batch (BatchT): The batch to take.

        Returns:
            list[BatchT]: Batches of exactly :attr:`size` rows, empty while there are still
            too few.
        """
        rows = num_rows(batch)
        if rows:
            self._pending.append(batch)
            self._rows += rows

        full = []
        while self._rows >= self.size:
            joined = concat(self._pending)
            full.append(slice_batch(joined, 0, self.size))

            rest = slice_batch(joined, self.size)
            self._rows = num_rows(rest)
            self._pending = [rest] if self._rows else []

        return full

    def take(self) -> None | BatchT:
        """Hand back whatever is left, however little, and empty the buffer.

        Args:
            None

        Returns:
            None | BatchT: The remaining rows as one batch, or None if there are none.
        """
        if not self._pending:
            return None

        joined = concat(self._pending)
        self._pending = []
        self._rows = 0
        return joined
