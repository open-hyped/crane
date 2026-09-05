"""The Slurm distributed backend.

Slurm is driven through its command line tools rather than through bindings. The bindings
available for Python are compiled against a specific Slurm release, which would tie a
:code:`pip install` of crane to the version running on the user's cluster; the four things
a run needs - submit an array, submit a dependent job, poll, cancel - are all available
from :code:`sbatch`, :code:`squeue` and :code:`scancel` with stable, parsable output.

Every call into Slurm goes through :func:`_run`, so replacing the command line with
bindings later is a change to this module alone.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import ClassVar

from .core.base import BackendView, DistributedBackend, JobState, RunSpec

logger = logging.getLogger(__name__)

_REQUIRED_COMMANDS = ("sbatch", "squeue", "scancel")
"""The Slurm tools a run cannot be driven without."""

_COMMENT_PREFIX = "crane:"
"""Marks a job as crane's, and names the run it belongs to.

Carried in Slurm's :code:`--comment` rather than in the job name, because the name is a
user-facing setting a run may well override. Nothing reads it back today - a run is
identified by its run directory - but it is what makes crane's jobs recognisable in
:code:`squeue` by eye.
"""


class SlurmNotAvailableError(RuntimeError):
    """Raised when a run is submitted from a machine that cannot talk to Slurm."""


def _require_slurm() -> None:
    """Check that this machine can drive Slurm at all.

    Raises:
        SlurmNotAvailableError: If any of the required tools is missing, naming them.
    """
    missing = [cmd for cmd in _REQUIRED_COMMANDS if shutil.which(cmd) is None]

    if missing:
        raise SlurmNotAvailableError(
            f"Slurm is not available here: {', '.join(missing)} not found on PATH. "
            f"Submitting a run requires a host that can talk to the cluster's controller, "
            f"which is usually a login or submission node. To run on this machine instead, "
            f"use `write` rather than `submit`."
        )


def _run(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run one Slurm command.

    Args:
        args (list[str]): The command and its arguments.
        check (bool): Whether a non-zero exit status is an error. False where a non-zero
            status is an ordinary answer, as it is when asking about a job that has already
            left the queue.

    Returns:
        subprocess.CompletedProcess: The finished command.

    Raises:
        RuntimeError: If the command failed and :code:`check` is set.
    """
    logger.debug(f"Running {' '.join(args)}.")
    proc = subprocess.run(args, capture_output=True, text=True)

    if check and proc.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(args)}` failed with exit code {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )

    return proc


@dataclass(frozen=True)
class Slurm(DistributedBackend):
    """Run the workload as a Slurm array job.

    The run becomes one array of :attr:`num_jobs` tasks, each processing its own share of
    the dataset's shards, followed - where the workload writes a dataset - by a single
    dependent job that writes the metadata once every task has succeeded. The dependent job
    is how a run finishes itself: Slurm leaves nothing of the submitting process running,
    so there is nobody else to do it.

    Example:
        .. code-block:: python

            writer.submit(ds, on=Slurm(num_jobs=16, partition="cpu", time="04:00:00"))
    """

    name: ClassVar[str] = "slurm"

    num_jobs: int
    """How many array tasks to split the run across."""

    partition: None | str = None
    """The partition to submit to. Defaults to the cluster's own default."""

    time: None | str = None
    """The walltime limit, in any format sbatch accepts, such as :code:`"04:00:00"`."""

    cpus_per_task: None | int = None
    """Cores to reserve per job. Also the default for the writer's :code:`num_proc`."""

    mem: None | str = None
    """Memory to reserve per job, such as :code:`"64GB"`."""

    account: None | str = None
    """The account to charge."""

    qos: None | str = None
    """The quality of service to request."""

    job_name: None | str = None
    """The job's name. Defaults to the run id.

    Only cosmetic: a run is identified by its :code:`--comment`, so renaming the job does
    not hide it from :func:`crane.dist.attach`.
    """

    logs_dir: None | str = None
    """Where to write job output. Defaults to :code:`logs/` inside the run directory."""

    work_dir: None | str = None
    """Where to put the run directory. Defaults to :code:`.crane/` inside the save
    directory, which is not available for a submitted consumer - there this is required."""

    max_concurrent: None | int = None
    """The most array tasks to let run at once. Defaults to no limit."""

    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to export before the job starts."""

    setup: list[str] = field(default_factory=list)
    """Shell lines to run before the job starts, such as loading a module or activating a
    virtual environment. The job has to be able to import crane."""

    sbatch_args: dict[str, str] = field(default_factory=dict)
    """Any other sbatch options, as :code:`{"--gres": "gpu:1"}`. The escape hatch for
    everything sbatch takes that is not named above."""

    requeue: bool = False
    """Whether Slurm may requeue a job that was preempted or whose node failed."""

    dry_run: bool = False
    """Render the scripts and print them instead of submitting. A wrong partition should
    cost a moment rather than a place in the queue."""

    @property
    def cpus_per_job(self) -> None | int:
        """The cores each job reserves, used to default the writer's :code:`num_proc`."""
        return self.cpus_per_task

    def _directives(self, spec: RunSpec, logs_dir: str, array: bool) -> list[str]:
        """Build the :code:`#SBATCH` lines shared by both scripts.

        Args:
            spec (RunSpec): The run being submitted.
            logs_dir (str): Where job output goes.
            array (bool): Whether this is the array of worker jobs, as opposed to the
                single finalize job.

        Returns:
            list[str]: The directives, in the order they will be written.
        """
        name = self.job_name or spec.run_id
        options: dict[str, None | str] = {
            "--job-name": name if array else f"{name}-finalize",
            # The marker `attach` looks for. Set on both jobs, so a run whose workers have
            # finished but whose metadata is still being written is still discoverable.
            "--comment": f"{_COMMENT_PREFIX}{spec.run_id}",
            "--partition": self.partition,
            "--time": self.time,
            "--cpus-per-task": str(self.cpus_per_task) if self.cpus_per_task else None,
            "--mem": self.mem,
            "--account": self.account,
            "--qos": self.qos,
        }

        if array:
            limit = f"%{self.max_concurrent}" if self.max_concurrent else ""
            options["--array"] = f"0-{spec.num_jobs - 1}{limit}"
            # %A is the array job's id and %a the task index, so every task gets its own
            # pair of files rather than interleaving into one.
            options["--output"] = os.path.join(logs_dir, "%A_%a.out")
            options["--error"] = os.path.join(logs_dir, "%A_%a.err")
        else:
            options["--output"] = os.path.join(logs_dir, "finalize-%j.out")
            options["--error"] = os.path.join(logs_dir, "finalize-%j.err")

        options |= self.sbatch_args

        directives = [f"#SBATCH {key}={value}" for key, value in options.items() if value]
        if self.requeue:
            directives.append("#SBATCH --requeue")

        return directives

    def _script(self, spec: RunSpec, logs_dir: str, command: list[str], array: bool) -> str:
        """Render a complete sbatch script.

        Args:
            spec (RunSpec): The run being submitted.
            logs_dir (str): Where job output goes.
            command (list[str]): The command the job runs.
            array (bool): Whether this is the array of worker jobs.

        Returns:
            str: The script, ready to be handed to sbatch.
        """
        lines = ["#!/bin/bash"]
        lines += self._directives(spec, logs_dir, array)
        lines += ["", "set -e", ""]
        # Announced before anything heavy happens, and unbuffered, so that a job which
        # dies during startup still says where it was and how far it got. Without this a
        # job killed while importing leaves an empty log, which looks identical to a job
        # that never started.
        lines += [
            "export PYTHONUNBUFFERED=1",
            f'echo "crane: run {spec.run_id} task ${{SLURM_ARRAY_TASK_ID:-finalize}} '
            f'on $(hostname) at $(date -Is)"',
            "",
        ]
        lines += self.setup
        lines += [f"export {key}={value}" for key, value in self.env.items()]
        lines += ["", " ".join(command), ""]
        return "\n".join(lines)

    def _sbatch(self, script: str, path: str, extra_args: list[str] = []) -> str:
        """Write a script out and submit it.

        Args:
            script (str): The rendered sbatch script.
            path (str): Where to keep it, inside the run directory, so that a submitted run
                can be inspected after the fact.
            extra_args (list[str]): Extra sbatch arguments, such as a dependency.

        Returns:
            str: The submitted job's id.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)

        # `--parsable` reduces sbatch's output to the bare job id, so there is no message
        # to parse and nothing to break when that message is reworded.
        proc = _run(["sbatch", "--parsable", *extra_args, path])
        job_id = proc.stdout.strip().split(";")[0]
        logger.info(f"Submitted slurm job {job_id} from {path}.")
        return job_id

    def submit(self, spec: RunSpec) -> list[str]:
        """Submit the array of worker jobs, and the finalize job that follows them.

        Args:
            spec (RunSpec): The run to start.

        Returns:
            list[str]: The array job's id, followed by the finalize job's id where there is
            one. Empty for a dry run, which submits nothing.
        """
        _require_slurm()

        logs_dir = self.logs_dir or os.path.join(spec.run_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        entrypoint = [spec.python, "-m", "crane.dist.core._entrypoint", spec.run_dir]
        worker_script = self._script(
            spec,
            logs_dir,
            command=[*entrypoint, "--job-index", "$SLURM_ARRAY_TASK_ID"],
            array=True,
        )
        finalize_script = self._script(
            spec, logs_dir, command=[*entrypoint, "--finalize"], array=False
        )

        if self.dry_run:
            print(f"# {spec.num_jobs} worker job(s) over {spec.num_shards} shard(s)")
            print(worker_script)
            if spec.needs_finalize:
                print("\n# finalize, after the workers succeed")
                print(finalize_script)
            return []

        job_ids = [self._sbatch(worker_script, os.path.join(spec.run_dir, "worker.sbatch"))]

        if spec.needs_finalize:
            # `afterok` rather than `afterany`: the metadata describes every shard, so
            # writing it after a job failed would describe a dataset that is missing rows.
            job_ids.append(
                self._sbatch(
                    finalize_script,
                    os.path.join(spec.run_dir, "finalize.sbatch"),
                    extra_args=[f"--dependency=afterok:{job_ids[0]}"],
                )
            )

        return job_ids

    def poll(self, spec: RunSpec, job_ids: list[str]) -> BackendView:
        """Ask Slurm which jobs of the run are still queued or running.

        Args:
            spec (RunSpec): The run to inspect.
            job_ids (list[str]): The ids :func:`submit` returned.

        Returns:
            BackendView: The array tasks still alive, and whether the finalize job is
            still to come.
        """
        _require_slurm()

        if not job_ids:  # pragma: not covered
            return BackendView(active_jobs={}, finalize_pending=False)

        return BackendView(
            active_jobs=self._active_array_tasks(job_ids[0]),
            finalize_pending=(len(job_ids) > 1) and self._is_active(job_ids[1]),
        )

    def _active_array_tasks(self, array_job_id: str) -> dict[int, JobState]:
        """The array tasks Slurm still knows about, by index.

        :code:`--array` expands the queue's collapsed ranges - a block of pending tasks is
        otherwise reported as a single :code:`1234_[3-15]` line - so that each task can be
        accounted for individually.

        Args:
            array_job_id (str): The array job to inspect.

        Returns:
            dict[int, JobState]: The tasks that are queued or running.
        """
        # A job that has left the queue makes squeue exit non-zero, which is an answer
        # rather than a failure: it means nothing of that job is alive any more.
        proc = _run(
            ["squeue", "--array", "--noheader", "--job", array_job_id, "--format=%K;%T"],
            check=False,
        )
        if proc.returncode != 0:
            return {}

        active = {}
        for line in proc.stdout.splitlines():
            index, _, state = line.strip().partition(";")
            if index.isdigit():
                # Anything the queue still lists that is not waiting to start is treated as
                # running, including the transient states such as COMPLETING: the
                # distinction that matters here is only "still alive".
                active[int(index)] = (
                    JobState.PENDING if state.strip() == "PENDING" else JobState.RUNNING
                )

        return active

    def _is_active(self, job_id: str) -> bool:
        """Whether Slurm still knows about a single job."""
        proc = _run(
            ["squeue", "--noheader", "--job", job_id, "--format=%T"],
            check=False,
        )
        return (proc.returncode == 0) and bool(proc.stdout.strip())

    def cancel(self, spec: RunSpec, job_ids: list[str]) -> None:
        """Cancel every job of the run, the finalize job included.

        Args:
            spec (RunSpec): The run to stop.
            job_ids (list[str]): The ids :func:`submit` returned.
        """
        _require_slurm()

        if job_ids:
            _run(["scancel", *job_ids], check=False)
            logger.info(f"Cancelled slurm job(s) {', '.join(job_ids)}.")

