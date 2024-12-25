"""Crane.

Lift (process) and place (write) data streams, seamlessly and in parallel.
"""

__all__ = [
    "Callback",
    "DatasetConsumer",
    "ShardingStrategy",
    "TqdmReporterCallback",
    "setup_logging",
]


from .core import Callback, DatasetConsumer, ShardingStrategy, TqdmReporterCallback
from .logging.setup import setup_logging
