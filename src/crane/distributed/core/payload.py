"""What a distributed job needs, and how it gets there.

This module defines the :class:`Payload` - the dataset and the writer or consumer that was
submitted - along with the functions that serialize it into the run directory and read it
back inside a job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import dill
from datasets import IterableDataset

if TYPE_CHECKING:  # pragma: not covered
    from ...core.consumer import DatasetConsumer
    from ...core.writer import BaseDatasetWriter


@dataclass(frozen=True)
class Payload(object):
    """Everything a job needs that is fixed for the whole run.

    Written once by :func:`crane.distributed.submit` and read by every worker job and by the
    finalize job, so that the three cannot drift apart.

    Dill rather than pickle for the same reason :class:`WorkerSetup` uses it: a transform is
    routinely a closure, a lambda or a partial over a local, which the standard pickler
    cannot take.

    Holds the writer as it was configured, never as it is mid-run. The
    :class:`ShardingController` a write builds owns per-process multiprocessing primitives,
    so each job creates its own as part of the ordinary write path rather than receiving
    one.
    """

    ds: IterableDataset
    """The dataset, transforms included."""

    target: BaseDatasetWriter | DatasetConsumer
    """The writer or consumer that was submitted."""

    num_jobs: int
    """How many jobs the run is split across."""

    finalizer: None | Callable[[Any], Any] = None
    """The consumer's workload. None when a writer was submitted; it supplies its own."""

    finalizer_batch_size: None | int = None
    """Batch size for :attr:`finalizer`."""

    finalizer_formatting: None | str = None
    """The data format :attr:`finalizer` expects."""


def dumps(payload: Payload) -> bytes:
    """Serialize a payload, having checked that it can be read back.

    Serializing is the expensive half of submitting a realistic workload - a payload
    carrying a tokenizer or a model is a large object graph, and `recurse=True` walks all
    of it - so the check and the write share one pass rather than taking one each.

    Args:
        payload (Payload): The payload to serialize.

    Returns:
        bytes: The serialized payload.

    Raises:
        TypeError: If the payload cannot be serialized, naming the underlying error.
    """
    try:
        blob = dill.dumps(payload, recurse=True)
        # Dumping is not proof of being readable, and the job would be the one to find out.
        dill.loads(blob)
    except Exception as e:
        raise TypeError(
            "The dataset, writer or workload cannot be sent to a distributed job. This "
            "usually means something in it closes over process-local state, such as a "
            f"database connection, a socket or an open generator. Original error: {e}"
        ) from e

    return blob


def dump(payload: Payload, path: str) -> None:
    """Write a payload into the run directory, having checked it can be read back.

    Args:
        payload (Payload): The payload to write.
        path (str): Where to write it.

    Raises:
        TypeError: If the payload cannot be serialized, naming the underlying error.
    """
    with open(path, "wb") as f:
        f.write(dumps(payload))


def load(path: str) -> Payload:
    """Read a payload back, inside a job.

    Args:
        path (str): The payload written at submission time.

    Returns:
        Payload: The workload to run.
    """
    with open(path, "rb") as f:
        return dill.load(f)


def check(payload: Payload) -> None:
    """Fail now if the payload cannot be serialized at all.

    A transform closing over a database connection, a socket or a live generator cannot be
    sent to a job. Finding that out here costs a moment; finding it out from a job that has
    already waited in a queue costs considerably more, and the traceback arrives in a log
    file rather than in the caller's terminal.

    **Note**: This is a serialization check, not a check that the payload will *behave*
    remotely. Dill accepts more than it should for our purposes - an open file handle is
    taken by path and reopened inside the job, and a :class:`threading.Lock` is recreated
    unlocked. Neither raises here, and both are still the wrong thing to close over.

    Args:
        payload (Payload): The payload about to be submitted.

    Raises:
        TypeError: If the payload cannot be serialized, naming the underlying error.
    """
    dumps(payload)
