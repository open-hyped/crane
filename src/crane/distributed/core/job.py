"""Which job of a distributed run this process belongs to.

Answers the question "am I job 3 of 8, and of which run?" from anywhere, without the
answer having to be threaded through every call in between - the same reason
:mod:`crane.core.worker` keeps the worker's rank in a module global rather than in an
argument. Code that never runs distributed sees :func:`get_job_info` return :code:`None`,
which is the local case and not an error.

The value is established once, at the top of a job, by :mod:`crane.distributed.core._entrypoint`
from the index the backend passed it on the command line. That is the only authority:
:func:`set_job_info` merely records what the entrypoint was told.

Because a run's processes fork or spawn workers of their own, the value is mirrored into
the environment as it is set. Environment variables are the one piece of process state that
survives both start methods and reaches a child on another node, so a worker inherits the
answer without the runners - which know nothing about distribution - having to carry it:

.. code-block:: bash

    CRANE_DISTRIBUTED_INDEX=3
    CRANE_DISTRIBUTED_NUM_JOBS=8
    CRANE_DISTRIBUTED_RUN_ID=out-650746

Setting those by hand is supported, and is how a single job of a failed run is reproduced
outside the scheduler. It also means a stale export makes a process believe it is
distributed, so anything that would silently change local output - the shard names, above
all - is driven by :func:`BaseDatasetWriter._set_shard_name` instead, which is passed
explicitly and cannot be switched on by the environment.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

INDEX_ENV_VAR = "CRANE_DISTRIBUTED_INDEX"
"""Environment variable holding the index of the job in the run."""

NUM_JOBS_ENV_VAR = "CRANE_DISTRIBUTED_NUM_JOBS"
"""Environment variable holding the number of jobs the run is split across."""

RUN_ID_ENV_VAR = "CRANE_DISTRIBUTED_RUN_ID"
"""Environment variable holding the id of the run."""


@dataclass(frozen=True)
class JobInfo(object):
    """Holds information about the distributed job the current process belongs to."""

    index: int
    """The index of this job, from zero."""

    num_jobs: int
    """The total number of jobs the run is split across."""

    run_id: str
    """The id of the run this job belongs to."""


_job_info: None | JobInfo = None


def get_job_info() -> None | JobInfo:
    """Retrieve the current process's distributed job information.

    Reads the module global if this process set it, and otherwise the environment, which is
    how a worker forked or spawned by a job finds the answer.

    Returns:
        None | JobInfo: The job information, or None if this process is not part of a
        distributed run.
    """
    global _job_info

    if _job_info is not None:
        return _job_info

    index = os.environ.get(INDEX_ENV_VAR)
    num_jobs = os.environ.get(NUM_JOBS_ENV_VAR)
    run_id = os.environ.get(RUN_ID_ENV_VAR)

    if index is None or num_jobs is None or run_id is None:
        return None

    try:
        _job_info = JobInfo(index=int(index), num_jobs=int(num_jobs), run_id=run_id)
    except ValueError:
        # Only reachable if something other than crane wrote these, in which case the
        # process is not part of a run and saying so beats failing at an arbitrary depth.
        # Warned rather than logged: the log formatter asks this question of every record,
        # so logging from here would call back into this function without end.
        warnings.warn(
            f"Ignoring the distributed job environment: `{INDEX_ENV_VAR}`={index!r} and "
            f"`{NUM_JOBS_ENV_VAR}`={num_jobs!r} are not both integers.",
            UserWarning,
            stacklevel=2,
        )
        return None

    return _job_info


def set_job_info(index: int, num_jobs: int, run_id: str) -> JobInfo:
    """Record which job of a run this process is.

    Called by the entrypoint of a job, before any work starts. Mirrored into the
    environment so that the processes this one goes on to start inherit it.

    Args:
        index (int): The index of this job, from zero.
        num_jobs (int): The total number of jobs the run is split across.
        run_id (str): The id of the run this job belongs to.

    Returns:
        JobInfo: The newly created job information.

    Raises:
        AssertionError: If job information is already set, preventing reassignment.
        ValueError: If the index does not address a job of the run.
    """
    global _job_info

    assert _job_info is None, "Job info already set."

    if not (0 <= index < num_jobs):
        raise ValueError(f"Job index {index} is out of range for {num_jobs} job(s).")

    _job_info = JobInfo(index=index, num_jobs=num_jobs, run_id=run_id)

    os.environ[INDEX_ENV_VAR] = str(index)
    os.environ[NUM_JOBS_ENV_VAR] = str(num_jobs)
    os.environ[RUN_ID_ENV_VAR] = run_id

    return _job_info


def reset_job_info() -> None:
    """Reset the distributed job information for the current process."""
    global _job_info
    _job_info = None

    for var in (INDEX_ENV_VAR, NUM_JOBS_ENV_VAR, RUN_ID_ENV_VAR):
        os.environ.pop(var, None)
