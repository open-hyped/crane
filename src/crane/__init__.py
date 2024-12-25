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
    "ShardingStrategy",
    "TqdmReporterCallback",
    "JsonDatasetWriter",
    "ArrowDatasetWriter",
    "setup_logging",
]

from typing import TYPE_CHECKING

from .core import Callback, DatasetConsumer, ShardingStrategy, TqdmReporterCallback

if TYPE_CHECKING:
    from .arrow import ArrowDatasetWriter
    from .json import JsonDatasetWriter

else:
    import importlib
    import sys

    def lazy_load(name: str) -> object:
        """Lazy loads module content based on attribute name."""
        module = {
            "JsonDatasetWriter": ".json",
            "ArrowDatasetWriter": ".arrow",
        }[name]
        module = importlib.import_module(module, package=__name__)
        return getattr(module, name)

    sys.modules[__name__].__getattr__ = lazy_load
