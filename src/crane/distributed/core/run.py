"""The handle a submitted run is driven through.

:func:`submit` starts a run and hands back a :class:`DistributedRun`. The handle can be
rebuilt from the run directory alone - see :func:`attach` - so nothing here requires the
process that submitted the run to still exist.
"""

from __future__ import annotations

import copy
import glob
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

from datasets import IterableDataset
from tqdm.auto import tqdm

from ...__version__ import __version__
from ...core.runners.base import FailurePolicy, ShardFailure, ShardProcessingError
from . import payload
from .base import SPEC_FILENAME, DistributedBackend, JobState, RunSpec, RunState, RunStatus
from .partition import num_jobs_for

if TYPE_CHECKING:  # pragma: not covered
    from ...core.consumer import DatasetConsumer
    from ...core.writer import BaseDatasetWriter

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 10.0
"""How often :func:`DistributedRun.wait` asks the scheduler for news, in seconds.

The scheduler's controller is a shared resource and a run is measured in minutes at best,
so polling harder buys nothing and costs the cluster.
"""


class RunNotFoundError(LookupError):
    """Raised when there is no run to attach to at a given location."""


class JobLostError(RuntimeError):
    """Raised when a job vanished without reporting anything.

    Distinct from :class:`ShardProcessingError`, which means the workload raised. A lost
    job was killed by something the workload never saw - a walltime, a preemption, an
    out-of-memory kill, a node failure - so there is no traceback to show and nothing in
    the user's code to fix.
    """


class DistributedRun(object):
    """A submitted run, and the things you can do to it once it is out of your hands.

    Combines two sources that are each authoritative about one thing: the scheduler knows
    which jobs are still alive, and the files the jobs write say what they actually did.
    Neither is trusted about the other - a job the scheduler has finished with that never
    wrote a result is exactly how a killed job is detected.
    """

    def __init__(self, spec: RunSpec, backend: DistributedBackend) -> None:
        """Initialize the handle.

        Args:
            spec (RunSpec): The run's description, as written to its run directory.
            backend (DistributedBackend): The backend that started it.
        """
        self._spec = spec
        self._backend = backend

    @property
    def run_id(self) -> str:
        """The run's identity."""
        return self._spec.run_id

    @property
    def run_dir(self) -> str:
        """Where the payload, per-job results and logs live, and what :func:`attach` takes."""
        return self._spec.run_dir

    @property
    def save_dir(self) -> None | str:
        """The dataset's output directory, or None for a submitted consumer."""
        return self._spec.save_dir

    @property
    def num_jobs(self) -> int:
        """How many worker jobs the run was split across."""
        return self._spec.num_jobs

    @property
    def failure_policy(self) -> FailurePolicy:
        """What this run does when the workload raises on a shard."""
        return FailurePolicy(self._spec.failure_policy)

    @property
    def job_ids(self) -> list[str]:
        """The backend's own job ids."""
        return list(self._spec.job_ids)

    def _read_result(self, job_index: int) -> None | dict[str, Any]:
        """Read what one job reported about itself, if it reported at all."""
        path = self._spec.job_result_path(job_index)
        if not os.path.exists(path):
            return None

        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError) as e:  # pragma: not covered
            logger.warning(f"Ignoring unreadable job result {path}: {e}")
            return None

    def status(self) -> RunStatus:
        """A snapshot of the run. Does not block.

        The scheduler is asked first and the result files second, which is the order that
        cannot produce a false "lost": a job writes its result before it exits, and exits
        before the scheduler forgets it, so anything the scheduler has already dropped has
        written whatever it was going to write.

        Returns:
            RunStatus: What every job is doing, and how far the run has got.
        """
        view = self._backend.poll(self._spec, self._spec.job_ids)

        jobs: dict[int, JobState] = {}
        failures: list[ShardFailure] = []
        lost: list[int] = []

        for job_index in range(self._spec.num_jobs):
            if job_index in view.active_jobs:
                jobs[job_index] = view.active_jobs[job_index]
                continue

            result = self._read_result(job_index)
            if result is None:
                jobs[job_index] = JobState.LOST
                lost.append(job_index)
                continue

            reported = [ShardFailure(**f) for f in result.get("failures", [])]
            failures.extend(reported)
            jobs[job_index] = JobState.FAILED if reported else JobState.COMPLETED

        return RunStatus(
            state=self._run_state(view.finalize_pending, jobs),
            jobs=jobs,
            shards_completed=len(glob.glob(os.path.join(self._spec.run_dir, "shards", "*"))),
            num_shards=self._spec.num_shards,
            failures=failures,
            lost_jobs=lost,
        )

    def _run_state(self, finalize_pending: bool, jobs: dict[int, JobState]) -> RunState:
        """Reduce the per-job states to a state for the run as a whole."""
        states = set(jobs.values())

        if states <= {JobState.PENDING}:
            return RunState.PENDING

        if states & {JobState.PENDING, JobState.RUNNING}:
            return RunState.RUNNING

        if states & {JobState.FAILED, JobState.LOST}:
            # A failed run stays failed even while the finalize job is technically still
            # queued: it depends on `afterok` and will never start.
            return RunState.FAILED

        return RunState.FINALIZING if finalize_pending else RunState.COMPLETED

    def wait(self, timeout: None | float = None, poll_interval: float = _POLL_INTERVAL) -> None:
        """Block until the run is finished, and raise if it did not succeed.

        Returns only once the metadata has been written too, so that what it returns to is
        a dataset that can be loaded rather than one that has merely finished computing.

        Acts on the run's failure policy while it waits; see :func:`_follow`.

        Args:
            timeout (None | float): Give up waiting after this many seconds. The run is not
                cancelled; only the waiting stops. Defaults to waiting indefinitely.
            poll_interval (float): How often to ask the scheduler for news, in seconds.

        Raises:
            ShardProcessingError: If the workload raised on any shard, carrying every
                failure from every job.
            JobLostError: If any job vanished without reporting.
            TimeoutError: If :code:`timeout` passed while the run was still going.
        """
        status = self._follow(timeout=timeout, poll_interval=poll_interval)

        if status.failures:
            # The same error a local run raises, with the same contents: every failure the
            # workload reported, from whichever job reported it.
            raise ShardProcessingError(status.failures)

        if status.lost_jobs:
            raise JobLostError(
                f"Job(s) {status.lost_jobs} of run {self.run_id} ended without reporting. "
                f"This is not a failure of the workload - the job was killed by something "
                f"outside it, such as the walltime, a preemption or an out-of-memory kill. "
                f"See `run.logs(job_index)`."
            )

    def watch(self, timeout: None | float = None, poll_interval: float = 1.0) -> None:
        """Follow the run with a progress bar, until it finishes.

        Counts shards completed across every job, which is the one measure that survives
        the trip off the machine. Per-job throughput stays in the per-job logs, where the
        numbers mean something.

        Watching a run drives it exactly as waiting for one does - the failure policy is
        acted on either way, because both are somebody following the run and it would be
        strange for the same failure to stop the cluster through one method and not the
        other. Unlike :func:`wait`, this returns quietly when the run has failed; ask
        :func:`status` or :func:`wait` what went wrong.

        Args:
            timeout (None | float): Give up following after this many seconds. The run is
                not cancelled; only the watching stops. Defaults to waiting indefinitely.
            poll_interval (float): How often to refresh, in seconds.

        Raises:
            TimeoutError: If :code:`timeout` passed while the run was still going.
        """
        with tqdm(total=self._spec.num_shards, unit="sh", desc=self.run_id) as pbar:

            def draw(status: RunStatus) -> None:
                done = sum(s is JobState.COMPLETED for s in status.jobs.values())
                pbar.set_postfix_str(f"{status.state.value}, {done}/{self.num_jobs} jobs")
                pbar.update(status.shards_completed - pbar.n)

            self._follow(timeout=timeout, poll_interval=poll_interval, on_status=draw)

    def _follow(
        self,
        timeout: None | float,
        poll_interval: float,
        on_status: None | Callable[[RunStatus], None] = None,
    ) -> RunStatus:
        """Poll until the run settles, acting on its failure policy as it goes.

        The one loop behind :func:`wait` and :func:`watch`, so that following a run means
        the same thing whichever of them is used.

        Args:
            timeout (None | float): Give up after this many seconds. The run is not
                cancelled; only the following stops. None waits indefinitely.
            poll_interval (float): How often to ask the scheduler for news, in seconds.
            on_status (None | Callable[[RunStatus], None]): Called with each snapshot, for
                a caller that wants to show progress.

        Returns:
            RunStatus: How the run ended - or how it stood when it was stopped, which is not
            the same thing; see below.

        Raises:
            TimeoutError: If :code:`timeout` passed while the run was still going.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        # What had gone wrong at the moment the run was stopped, if it was. Kept because
        # cancelling the siblings makes them look lost too, and reporting the jobs this
        # handle killed alongside the one that actually failed would bury the real fault.
        stopped_on: None | RunStatus = None

        while True:
            status = self.status()

            if on_status is not None:
                on_status(status)

            if (stopped_on is None) and self._should_stop_the_rest(status):
                logger.error(
                    f"Run {self.run_id} failed on {len(status.failures)} shard(s) and the "
                    f"failure policy is {self.failure_policy.value}; stopping the "
                    f"remaining jobs."
                )
                stopped_on = status
                self.cancel()

            if status.state in (RunState.COMPLETED, RunState.FAILED):
                break

            if (deadline is not None) and (time.monotonic() >= deadline):
                raise TimeoutError(
                    f"Run {self.run_id} was still {status.state.value} after {timeout}s. "
                    f"It has not been cancelled; attach to it again to keep waiting."
                )

            time.sleep(poll_interval)

        # A run that was stopped is reported as it was when the decision was taken, not as
        # it looks afterwards.
        return stopped_on if stopped_on is not None else status

    def _should_stop_the_rest(self, status: RunStatus) -> bool:
        """Whether a failure means the jobs still running should be stopped.

        :attr:`FailurePolicy.FAIL_FAST` says a run whose workload raised has not produced
        the dataset that was asked for, so there is nothing to be gained by letting the rest
        of the cluster keep working on it. Locally the runner stops its own workers; the
        jobs of a distributed run have no such connection to each other, and stopping them
        is the handle's to do.

        :attr:`FailurePolicy.SKIP_SHARD` means the opposite - the failed shards are expected
        losses - so the other jobs carry on.

        **Note**: this only happens while something is following the run. One left alone
        after :func:`submit` keeps going until its jobs finish on their own.

        Args:
            status (RunStatus): The snapshot to judge.

        Returns:
            bool: Whether to cancel the run.
        """
        if self.failure_policy is not FailurePolicy.FAIL_FAST:
            return False

        # A lost job is not counted here: it is not the workload failing, and a run that
        # loses one job to a preemption may still be worth letting finish.
        return bool(status.failures)

    def cancel(self) -> None:
        """Stop every job of the run, the finalize job included."""
        self._backend.cancel(self._spec, self._spec.job_ids)

    def logs(self, job_index: int, stream: str = "out") -> str:
        """Read one job's output.

        The first place to look when a job was lost, since whatever killed it left nothing
        else behind.

        Args:
            job_index (int): The job to read.
            stream (str): Either :code:`"out"` or :code:`"err"`.

        Returns:
            str: What the job wrote, or a note saying the log could not be found.
        """
        pattern = os.path.join(self._spec.run_dir, "logs", f"*_{job_index}.{stream}")
        matches = sorted(glob.glob(pattern))

        if not matches:
            return f"No {stream} log found for job {job_index} (looked for {pattern})."

        with open(matches[-1], encoding="utf-8", errors="replace") as f:
            return f.read()

    def __repr__(self) -> str:
        """Identify the run and where its output goes."""
        return (
            f"DistributedRun(run_id={self.run_id!r}, num_jobs={self.num_jobs}, "
            f"save_dir={self.save_dir!r})"
        )


def submit(
    ds: IterableDataset,
    target: BaseDatasetWriter | DatasetConsumer,
    backend: DistributedBackend,
    save_dir: None | str,
    needs_finalize: bool,
    finalizer: None | Callable[[Any], Any] = None,
    finalizer_batch_size: None | int = None,
    finalizer_formatting: None | str = None,
) -> DistributedRun:
    """Start a distributed run.

    Shared by :func:`BaseDatasetWriter.submit` and :func:`DatasetConsumer.submit`. Neither
    forwards to the other - their payloads differ, since only a writer has an output
    directory to prepare and metadata to write - but the mechanics of getting a workload
    onto a cluster are the same for both.

    Everything checkable cheaply is checked here rather than in a job, so that the ordinary
    mistakes surface in the caller's terminal instead of in a log file an hour later.

    Args:
        ds (IterableDataset): The dataset to process.
        target (BaseDatasetWriter | DatasetConsumer): What to process it with.
        backend (DistributedBackend): Where to run.
        save_dir (None | str): The dataset's output directory, if there is one.
        needs_finalize (bool): Whether a finalize job must follow the workers.
        finalizer (None | Callable[[Any], Any]): A submitted consumer's workload.
        finalizer_batch_size (None | int): Batch size for :code:`finalizer`.
        finalizer_formatting (None | str): The format :code:`finalizer` expects.

    Returns:
        DistributedRun: A handle on the submitted run.

    Raises:
        ValueError: If the run has nowhere to keep its working files.
    """
    num_jobs = num_jobs_for(ds, backend.num_jobs)
    run_id = _make_run_id(target, save_dir)
    run_dir = os.path.join(_work_dir(backend, save_dir), run_id)

    os.makedirs(os.path.join(run_dir, "jobs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "shards"), exist_ok=True)

    # The job runs with as many processes as the backend reserved cores, unless the caller
    # said otherwise. Configured onto a copy so that the caller's own object is untouched.
    target = copy.copy(target)
    if target._num_proc is None:
        target._num_proc = backend.cpus_per_job

    data = payload.Payload(
        ds=ds,
        target=target,
        num_jobs=num_jobs,
        finalizer=finalizer,
        finalizer_batch_size=finalizer_batch_size,
        finalizer_formatting=finalizer_formatting,
    )
    payload.check(data)

    spec = RunSpec(
        run_id=run_id,
        run_dir=run_dir,
        num_jobs=num_jobs,
        num_shards=ds.n_shards,
        save_dir=save_dir,
        # Taken from the submitting process, so that a job lands in the environment the
        # dataset was built in rather than whatever python the compute node happens to have.
        python=sys.executable,
        needs_finalize=needs_finalize,
        # A writer and a consumer both carry one; a target that somehow does not gets the
        # same default they do.
        failure_policy=getattr(target, "_failure_policy", FailurePolicy.FAIL_FAST).value,
        backend=backend.to_dict(),
        crane_version=__version__,
    )

    payload.dump(data, spec.payload_path)
    spec.save()

    # Saved twice: the jobs need the spec in order to start, and the ids only exist once
    # they have been submitted.
    spec = replace(spec, job_ids=backend.submit(spec))
    spec.save()

    logger.info(
        f"Submitted run {run_id} as {num_jobs} job(s) over {ds.n_shards} shard(s); "
        f"run directory {run_dir}."
    )
    return DistributedRun(spec, backend)


def _work_dir(backend: DistributedBackend, save_dir: None | str) -> str:
    """Where the run keeps its working files.

    Raises:
        ValueError: If there is neither a backend work directory nor an output directory.
    """
    work_dir = getattr(backend, "work_dir", None)

    if work_dir is not None:
        return work_dir

    if save_dir is None:
        raise ValueError(
            "A distributed consumer writes no dataset, so there is no output directory to "
            "keep the run's working files beside. Pass `work_dir` to the backend."
        )

    return os.path.join(save_dir, ".crane")


def _make_run_id(target: Any, save_dir: None | str) -> str:
    """Build a run id that says what the run is and is unique to this submission.

    The readable half is the output directory's name, or the target's type where there is
    no output directory. The random half is what keeps a resubmission of the same work from
    colliding with the run it repeats.
    """
    stem = os.path.basename(save_dir.rstrip("/")) if save_dir else type(target).__name__
    stem = "".join(c if (c.isalnum() or c in "-_") else "-" for c in stem).strip("-")
    return f"{stem or 'crane'}-{uuid.uuid4().hex[:6]}"


def attach(run_dir: str) -> DistributedRun:
    """Rebuild a handle on a run that was submitted elsewhere.

    Everything needed is in the run directory, so this works from any process on any
    machine that can see it - which is the point of :func:`submit` returning rather than
    blocking.

    **Note**: The run directory is printed when a run is submitted, and is
    :code:`<save_dir>/.crane/<run_id>` unless the backend was given a :code:`work_dir`.

    Args:
        run_dir (str): Where the run keeps its files.

    Returns:
        DistributedRun: A handle on the run.

    Raises:
        RunNotFoundError: If there is no run there.
    """
    if not os.path.exists(os.path.join(run_dir, SPEC_FILENAME)):
        raise RunNotFoundError(f"{run_dir} does not look like a crane run directory.")

    spec = RunSpec.load(run_dir)
    return DistributedRun(spec, DistributedBackend.from_dict(spec.backend))
