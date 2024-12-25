"""Base Module for dataset consumption and writing.

This module provides high-level functionality for processing large datasets using a dynamic
multiprocessing system. It includes utilities for managing worker processes, tracking progress,
and implementing custom dataset writers.
"""

__all__ = [
    "ShardingStrategy",
    "DatasetConsumer",
    "BaseDatasetWriter",
    "Callback",
    "TqdmReporterCallback",
]

from .callbacks.base import Callback
from .callbacks.tqdm_reporter import TqdmReporterCallback
from .consumer import DatasetConsumer
from .sharding import ShardingStrategy
from .writer import BaseDatasetWriter
