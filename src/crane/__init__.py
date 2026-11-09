"""Crane.

Lift (process) and place (write) data streams, seamlessly and in parallel.
"""

from .__version__ import __version__, __version_tuple__  # noqa: F401

# isort: off
# keep logging setup at beginning of the file
# to run the setup hook first
from .logging.setup import setup_logging

# isort: on

__all__ = [
    "Callback",
    "DatasetConsumer",
    "FailurePolicy",
    "ShardFailure",
    "ShardProcessingError",
    "ShardingStrategy",
    "TqdmReporterCallback",
    "JsonDatasetWriter",
    "ArrowDatasetWriter",
    "ParquetDatasetWriter",
    "setup_logging",
]

from typing import TYPE_CHECKING

from .core import (
    Callback,
    DatasetConsumer,
    FailurePolicy,
    ShardFailure,
    ShardingStrategy,
    ShardProcessingError,
    TqdmReporterCallback,
)

if TYPE_CHECKING:  # pragma: not covered
    from .arrow import ArrowDatasetWriter
    from .json import JsonDatasetWriter
    from .parquet import ParquetDatasetWriter

else:
    import importlib
    import sys

    LAZY_MODULES = {
        "JsonDatasetWriter": ".json",
        "ArrowDatasetWriter": ".arrow",
        "ParquetDatasetWriter": ".parquet",
    }

    def lazy_load(name: str) -> object:
        """Lazy loads module content based on attribute name.

        Raises `AttributeError` for any other name, as a module `__getattr__` must: `getattr`
        with a default and `hasattr` rely on it, and so does code that probes every loaded
        module for a name, like skops looking for scipy's ufuncs.
        """
        module = LAZY_MODULES.get(name)
        if module is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        return getattr(importlib.import_module(module, package=__name__), name)

    sys.modules[__name__].__getattr__ = lazy_load
