"""This module provides a logging formatter that adds worker rank information to log messages."""
import logging

from ..core.worker import get_worker_info


class RankAwareFormatter(logging.Formatter):
    """Rank Aware Logging Formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Adds a rank info to log messages based on worker information."""
        info = get_worker_info()
        record.rank_prefix = "" if info is None else f"[Rank {info.rank}] "
        return super().format(record)
