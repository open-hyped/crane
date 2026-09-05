"""Base types for distributed backends.

This module defines the interface a distributed backend implements, and the small value
types that travel between a backend and the run handle that drives it. Nothing here knows
anything about datasets: a backend's job is to start :code:`num_jobs` copies of a command
and report on them afterwards.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, ClassVar

SPEC_FILENAME = "spec.json"
"""Name of the run description written into the run directory."""

PAYLOAD_FILENAME = "payload.dill"
"""Name of the serialized workload written into the run directory."""


class JobState(str, Enum):
    """The state of a single job of a distributed run."""

    PENDING = "pending"
    """Queued, but not started."""

    RUNNING = "running"
    """Started and still alive."""

    COMPLETED = "completed"
    """Finished, and reported that it processed its shards."""

    FAILED = "failed"
    """Finished, and reported that it did not do all of its work.

    Either the workload raised on one or more shards, or the job failed before it could
    process any - see :attr:`RunStatus.failures` and :attr:`RunStatus.errors`.
    """

    LOST = "lost"
    """Gone from the scheduler without reporting anything.

    The job died in a way its own code never saw - a walltime, a preemption, an OOM kill,
    a node failure. There is no :class:`ShardFailure` for it, because nothing inside the
    job was running when it went.
    """


class RunState(str, Enum):
    """The state of a distributed run as a whole."""

    PENDING = "pending"
    """No job has started yet."""

    RUNNING = "running"
    """At least one job is alive."""

    FINALIZING = "finalizing"
    """Every job is done; the metadata is being written."""

    COMPLETED = "completed"
    """Finished. The dataset is written and loadable."""

    FAILED = "failed"
    """Finished, but at least one job failed or was lost."""


@dataclass(frozen=True)
class BackendView:
    """What the scheduler knows about a run.

    Deliberately coarse: the scheduler is authoritative about which jobs are still alive
    and about nothing else. What each job actually did is read from the files it wrote;
    see :class:`DistributedRun.status`.
    """

    active_jobs: dict[int, JobState]
    """The worker jobs the scheduler still knows about, by index.

    Only ever :attr:`JobState.PENDING` or :attr:`JobState.RUNNING` - what a finished job
    actually did is read from the file it wrote, not from the scheduler.
    """

    finalize_pending: bool
    """Whether a finalize job exists and has not finished yet."""


@dataclass(frozen=True)
class RunSpec:
    """The description of a run, as written to the run directory.

    Everything needed to rebuild a handle for a run in another process, days later, is
    here; see :func:`crane.distributed.attach`.
    """

    run_id: str
    """The run's identity, unique per submission."""

    run_dir: str
    """Where the payload, the per-job results and the logs live."""

    num_jobs: int
    """How many worker jobs the run was split across."""

    num_shards: int
    """How many shards of the dataset the run covers, across all jobs."""

    save_dir: None | str
    """The dataset's output directory, or None for a submitted consumer."""

    python: str
    """The interpreter the jobs run, taken from the submitting process."""

    needs_finalize: bool
    """Whether a finalize job follows the workers.

    False for a submitted consumer, which writes no dataset metadata and therefore has
    nothing that needs a view of every shard at once.
    """

    backend: dict[str, Any]
    """The backend, as :func:`DistributedBackend.to_dict` rendered it."""

    failure_policy: str = "fail_fast"
    """What the run does when the workload raises on a shard.

    Taken from the writer or consumer that was submitted. Recorded here because the decision
    it drives - whether the other jobs carry on - belongs to the handle watching the run, and
    the handle has only the spec to go on.
    """

    job_ids: list[str] = field(default_factory=list)
    """The backend's own job ids, written once the run has been submitted.

    Recorded rather than recovered from the scheduler later: a finished run is gone from
    the queue, and these are still what identifies it.
    """

    crane_version: str = ""
    """The version of crane that submitted the run, for diagnosing a mismatch."""

    def save(self) -> None:
        """Write the spec into its own run directory."""
        with open(os.path.join(self.run_dir, SPEC_FILENAME), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, run_dir: str) -> RunSpec:
        """Read the spec back out of a run directory.

        Args:
            run_dir (str): The run directory to read.

        Returns:
            RunSpec: The spec that was written at submission time.
        """
        with open(os.path.join(run_dir, SPEC_FILENAME), encoding="utf-8") as f:
            return cls(**json.load(f))

    @property
    def payload_path(self) -> str:
        """Path of the serialized workload."""
        return os.path.join(self.run_dir, PAYLOAD_FILENAME)

    def job_result_path(self, job_index: int) -> str:
        """Path of one job's result file.

        Args:
            job_index (int): The job to locate.

        Returns:
            str: Where that job writes what it did.
        """
        return os.path.join(self.run_dir, "jobs", f"{job_index}.json")

    @property
    def finalize_result_path(self) -> str:
        """Path of the finalize job's result file.

        Beside the worker results and read the same way: a finalize job that ends without
        writing one did not write the metadata either, whatever the scheduler says about it.
        """
        return os.path.join(self.run_dir, "jobs", "finalize.json")

    def shard_marker_path(self, job_index: int, shard_id: int) -> str:
        """Path of the marker a job drops when it completes a shard.

        The job index is part of the name: shard ids are only unique within a job, since
        the job index lives in the shard's file name instead, so two jobs both counting
        from zero would otherwise overwrite each other's markers.

        Args:
            job_index (int): The job that completed the shard.
            shard_id (int): The shard it completed, as numbered within that job.

        Returns:
            str: Where the marker goes.
        """
        return os.path.join(self.run_dir, "shards", f"{job_index}-{shard_id}")


class DistributedBackend(ABC):
    """A description of where a run executes.

    A backend holds no run state and does nothing until a run is handed to it, which is
    what lets the backends be interchangeable in :code:`submit(ds, on=...)`.

    Subclasses are expected to be frozen dataclasses, so that :func:`to_dict` and
    :func:`from_dict` can round-trip them through the run's spec file without any
    per-backend serialization code.
    """

    name: ClassVar[str]
    """The backend's identifier, as recorded in the spec."""

    _REGISTRY: ClassVar[dict[str, type[DistributedBackend]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register the subclass so that a spec can name it."""
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None) is not None:
            DistributedBackend._REGISTRY[cls.name] = cls

    num_jobs: int
    """How many jobs to split the run across.

    Declared here rather than as an abstract property: subclasses are dataclasses, and a
    field would not satisfy an abstract property - :class:`ABCMeta` would still find the
    abstract descriptor on the class and refuse to instantiate it.
    """

    @property
    def cpus_per_job(self) -> None | int:
        """The cores each job reserves, used to default the writer's :code:`num_proc`.

        Returns None when the backend does not reserve cores, in which case the local
        default applies.
        """
        return None

    @abstractmethod
    def submit(self, spec: RunSpec) -> list[str]:
        """Start the run.

        Args:
            spec (RunSpec): The run to start.

        Returns:
            list[str]: The backend's own job ids, for reporting and cancellation.
        """
        ...

    @abstractmethod
    def poll(self, spec: RunSpec, job_ids: list[str]) -> BackendView:
        """Ask the scheduler what is still alive.

        Args:
            spec (RunSpec): The run to inspect.
            job_ids (list[str]): The ids :func:`submit` returned.

        Returns:
            BackendView: Which jobs the scheduler still knows about.
        """
        ...

    @abstractmethod
    def cancel(self, spec: RunSpec, job_ids: list[str]) -> None:
        """Stop the run.

        Args:
            spec (RunSpec): The run to stop.
            job_ids (list[str]): The ids :func:`submit` returned.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Render the backend into the spec file."""
        return {"name": type(self).name} | asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> DistributedBackend:
        """Rebuild a backend from a spec file.

        Args:
            data (dict[str, Any]): What :func:`to_dict` wrote.

        Returns:
            DistributedBackend: The backend that submitted the run.

        Raises:
            ValueError: If the spec names a backend this installation does not have.
        """
        data = dict(data)
        name = data.pop("name")

        if name not in DistributedBackend._REGISTRY:
            raise ValueError(
                f"The run was submitted with the {name!r} backend, which is not available "
                f"here. Known backends: {sorted(DistributedBackend._REGISTRY)}."
            )

        return DistributedBackend._REGISTRY[name](**data)


@dataclass(frozen=True)
class RunStatus:
    """A snapshot of a run, combining the scheduler's view with what the jobs reported."""

    state: RunState
    """The run as a whole."""

    jobs: dict[int, JobState]
    """The state of each worker job, by index."""

    shards_completed: int
    """Shards finished so far, across every job."""

    num_shards: int
    """Shards the run covers in total."""

    failures: list[Any] = field(default_factory=list)
    """Every :class:`ShardFailure` reported by every job, in the order they were read."""

    errors: dict[int, str] = field(default_factory=dict)
    """Tracebacks from jobs that failed before processing any shard, by job index.

    Distinct from :attr:`failures`, which is the workload raising on a shard the job did
    reach. A job in here produced none of its shards, so its share of the dataset is
    missing entirely.
    """

    finalize_error: None | str = None
    """Why the dataset's metadata was not written, if it was not.

    A run whose shards all exist is still not loadable until the finalize job has run, so
    this is a failure of the run even though every worker succeeded.
    """

    lost_jobs: list[int] = field(default_factory=list)
    """Jobs that vanished without reporting - see :attr:`JobState.LOST`."""

    def __str__(self) -> str:
        """One-line summary."""
        return (
            f"{self.state.value} - {self.shards_completed}/{self.num_shards} shards, "
            f"{sum(s is JobState.COMPLETED for s in self.jobs.values())}/{len(self.jobs)} "
            f"jobs done"
        )
