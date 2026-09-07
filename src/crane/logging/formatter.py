"""This module provides a logging formatter that says which process wrote each message.

A line is prefixed with the worker rank, the distributed job, or both, depending on which
of the two the writing process is part of.
"""
import logging

from ..core.worker import get_worker_info
from ..distributed.core.job import get_job_info


class RankAwareFormatter(logging.Formatter):
    """Rank Aware Logging Formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Adds rank info to log messages based on worker and distributed job information.

        A distributed run's jobs each write their own log file, and every one of them numbers
        its workers from zero, so the rank alone does not say which line came from where once
        the files are read together.

        Counted from one and written against the total, so that a reader can tell how far
        through a run a line is without knowing how the run was configured: the last job of
        eight reads :code:`8/8` rather than :code:`7`.
        """
        job = get_job_info()
        worker = get_worker_info()

        parts = []
        if job is not None:
            parts.append(f"Job {job.index + 1}/{job.num_jobs}")
        if worker is not None:
            parts.append(f"Rank {worker.rank + 1}/{worker.num_workers}")

        record.rank_prefix = f"[{' '.join(parts)}] " if parts else ""
        return super().format(record)
