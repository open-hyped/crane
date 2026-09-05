"""Distributed Runs.

Spreads a run across the jobs of a cluster, where :code:`num_proc` spreads it across the
processes of one machine. Each job then runs exactly the runner crane uses locally, over
its own share of the dataset's shards.

The backend is passed to :func:`submit`, which returns as soon as the work is queued:

.. code-block:: python

    from crane import ArrowDatasetWriter
    from crane.distributed import Slurm

    writer = ArrowDatasetWriter("/shared/out", overwrite=True)
    run = writer.submit(ds, on=Slurm(num_jobs=16, partition="cpu", cpus_per_task=8))
    run.wait()

A run can be picked up again from another process, or another machine, with
:func:`attach` and the run directory it printed when it was submitted.

The backends live at this level, beside this module; everything they are built from is in
:mod:`crane.distributed.core`.
"""

__all__ = [
    "BackendView",
    "DistributedBackend",
    "DistributedRun",
    "JobLostError",
    "JobState",
    "RunNotFoundError",
    "RunSpec",
    "RunState",
    "RunStatus",
    "Slurm",
    "SlurmNotAvailableError",
    "attach",
]

from .core import (
    BackendView,
    DistributedBackend,
    DistributedRun,
    JobLostError,
    JobState,
    RunNotFoundError,
    RunSpec,
    RunState,
    RunStatus,
    attach,
)
from .slurm import Slurm, SlurmNotAvailableError
