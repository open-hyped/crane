"""Base Module for distributed runs.

This module provides the machinery a distributed run is built from: the description of a
run and the interface a backend implements, how a dataset's shards are divided between
jobs, what travels to a job and how, the handle a submitted run is driven through, and the
entry point a job actually executes.

The backends themselves are not here. They are the user-facing half of :mod:`crane.dist`
and live beside it, in the same way the concrete writers live beside :mod:`crane.core`.
"""

__all__ = [
    "BackendView",
    "DistributedBackend",
    "DistributedRun",
    "JobLostError",
    "JobState",
    "Payload",
    "RunNotFoundError",
    "RunSpec",
    "RunState",
    "RunStatus",
    "attach",
    "num_jobs_for",
    "select_shards",
    "submit",
]

from .base import BackendView, DistributedBackend, JobState, RunSpec, RunState, RunStatus
from .partition import num_jobs_for, select_shards
from .payload import Payload
from .run import DistributedRun, JobLostError, RunNotFoundError, attach, submit
