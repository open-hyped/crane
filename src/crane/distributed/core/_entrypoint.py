"""What a distributed job actually runs.

Two modes, one module:

.. code-block:: bash

    python -m crane.distributed._entrypoint <run_dir> --job-index <i>   # one job's shards
    python -m crane.distributed._entrypoint <run_dir> --finalize        # the metadata, once

Kept separate from :func:`write` and :func:`submit` so that each of the three does one
thing: run here, run locally, or launch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from typing import Any

from datasets import IterableDataset

from ...core.callbacks.base import Callback
from ...core.monitor import ProgressMonitor
from ...core.runners.base import ShardProcessingError
from ...logging.setup import setup_logging
from . import payload
from .base import RunSpec
from .partition import select_shards

logger = logging.getLogger(__name__)


class ShardMarkerCallback(Callback):
    """Drops a marker file in the run directory for every shard a job completes.

    The markers are what :func:`DistributedRun.status` counts, so a run's progress can be
    read from any machine that can see the run directory, without asking the scheduler or
    the jobs anything.
    """

    def __init__(self, spec: RunSpec, job_index: int) -> None:
        """Initialize the callback.

        Args:
            spec (RunSpec): The run this job belongs to.
            job_index (int): The index of this job, which is part of the marker's name.
                Shard ids are only unique within a job - the job index lives in the shard's
                file name instead - so markers named after the id alone would have the jobs
                overwriting each other's.
        """
        self._spec = spec
        self._job_index = job_index

    def on_shard_completed(self, monitor: ProgressMonitor, shard_id: int) -> None:
        """Record that a shard is finished.

        A failure to write the marker is logged rather than raised: this is progress
        reporting, and losing a tick of a progress bar is not worth losing the shard's data
        over.

        Args:
            monitor (ProgressMonitor): The monitor tracking the job's progress.
            shard_id (int): The shard that was completed.
        """
        path = self._spec.shard_marker_path(self._job_index, shard_id)
        try:
            with open(path, "w"):
                pass
        except OSError as e:  # pragma: not covered
            logger.warning(f"Could not write the progress marker {path}: {e}")

    def on_start(self, monitor: ProgressMonitor, ds: IterableDataset) -> None:
        """Make sure the markers have somewhere to go."""
        os.makedirs(os.path.dirname(self._spec.shard_marker_path(0, 0)), exist_ok=True)


def _report(spec: RunSpec, job_index: int, failures: list[Any], error: None | str) -> None:
    """Write what this job did, for the run's handle to read.

    A job that ends without writing one of these is how the handle detects a job that was
    killed by something outside the workload - a walltime, a preemption, an out-of-memory
    kill - since nothing inside the job is running to report that.

    Args:
        spec (RunSpec): The run this job belongs to.
        job_index (int): The job reporting.
        failures (list[Any]): The shard failures the workload reported, if any.
        error (None | str): A traceback for a failure that was not a shard failure.
    """
    path = spec.job_result_path(job_index)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # `ShardFailure` is a NamedTuple, so it renders to a dict the handle can rebuild it
    # from without either side knowing the field order.
    result = {
        "job_index": job_index,
        "failures": [f._asdict() for f in failures],
        "error": error,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def run_worker(run_dir: str, job_index: int) -> int:
    """Process one job's share of the dataset.

    Writes shards and nothing else. The metadata files each need a view of every shard at
    once, so they are written afterwards by :func:`run_finalize`.

    Args:
        run_dir (str): The run directory holding the payload.
        job_index (int): Which share of the shards to take.

    Returns:
        int: The process exit code.
    """
    spec = RunSpec.load(run_dir)
    data = payload.load(spec.payload_path)
    marker = ShardMarkerCallback(spec, job_index)

    logger.info(f"Job {job_index} of {spec.num_jobs} starting for run {spec.run_id}.")

    try:
        if spec.save_dir is not None:
            data.target._write_splits(
                data.ds, job_index=job_index, num_jobs=data.num_jobs, callbacks=[marker]
            )
        else:
            # A submitted consumer writes no dataset, so it takes its shards here rather
            # than inside a write path it does not have.
            data.target.add_callback(marker)
            data.target.consume(
                select_shards(data.ds, data.num_jobs, job_index),
                finalizer=data.finalizer,
                batch_size=data.finalizer_batch_size,
                formatting=data.finalizer_formatting,
            )

    except ShardProcessingError as e:
        # The workload raised on one or more shards. Reported rather than re-raised as a
        # traceback, so that the run's handle can rebuild the same error the local path
        # raises, carrying every failure from every job.
        logger.error(f"Job {job_index} finished with {len(e.failures)} shard failure(s).")
        _report(spec, job_index, failures=e.failures, error=None)
        return 1

    except Exception:
        logger.exception(f"Job {job_index} failed before it could process any shard.")
        _report(spec, job_index, failures=[], error=traceback.format_exc())
        return 1

    _report(spec, job_index, failures=[], error=None)
    logger.info(f"Job {job_index} finished.")
    return 0


def run_finalize(run_dir: str) -> int:
    """Write the metadata that needs every shard to exist.

    Runs once, after every worker job has succeeded. Deliberately the same call the local
    write path makes, so that a writer which gets this right locally gets it right
    distributed.

    Args:
        run_dir (str): The run directory holding the payload.

    Returns:
        int: The process exit code.
    """
    spec = RunSpec.load(run_dir)
    data = payload.load(spec.payload_path)

    logger.info(f"Finalizing run {spec.run_id} in {spec.save_dir}.")

    try:
        data.target._finalize_splits(data.ds)
    except Exception:
        logger.exception(f"Finalizing run {spec.run_id} failed.")
        return 1

    logger.info(f"Run {spec.run_id} finalized.")
    return 0


def main(argv: None | list[str] = None) -> int:
    """Parse the job's arguments and run it.

    Args:
        argv (None | list[str]): The arguments, for testing. Defaults to the real ones.

    Returns:
        int: The process exit code.
    """
    parser = argparse.ArgumentParser(description="Run one job of a distributed crane run.")
    parser.add_argument("run_dir", help="The run directory holding the payload.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-index", type=int, help="Process this job's share of the shards.")
    group.add_argument("--finalize", action="store_true", help="Write the dataset metadata, once.")
    args = parser.parse_args(argv)

    # A job has no terminal to watch, so default it to a level worth keeping in the log.
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO").upper(), log_file=os.getenv("LOG_FILE"))

    if args.finalize:
        return run_finalize(args.run_dir)
    return run_worker(args.run_dir, args.job_index)


if __name__ == "__main__":
    sys.exit(main())
