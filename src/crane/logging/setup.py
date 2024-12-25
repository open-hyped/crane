"""This module provides a utility function to configure logging for the application.

The logging setup supports both console and file logging, with a customizable
formatter that integrates worker rank information for distributed environments.
"""

import logging
import logging.config
import os


def setup_logging(level: int | str, log_file: None | str = None) -> None:
    """Set up the logging configuration.

    Args:
        level (None | int | str, optional): The logging level.
        log_file (None | str, optional): The file path for logging output. If set to
            :code:`None` (default) no log file is created.

    **Note**: By default, this function is called before execution, with the values
        for :code:`level` and :code:`log_file` taken from the environment variables
        :code:`LOG_LEVEL` and :code:`LOG_FILE`, respectively. If these environment
        variables are not set, the default logging level is :code:`"WARNING"`, and no
        log file will be created.
    """
    logging_config = {
        "version": 1,
        "formatters": {
            "custom": {
                "()": "crane.logging.formatter.RankAwareFormatter",
                "format": "%(rank_prefix)s[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "custom",
            },
        },
        "loggers": {
            "crane": {
                "handlers": ["console"],
                "level": level,
            },
        },
    }

    # add file handler
    if log_file is not None:
        logging_config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "custom",
            "filename": log_file,
            "mode": "a",
        }
        logging_config["loggers"]["crane"]["handlers"].append("file")

    logging.config.dictConfig(logging_config)


# setup logging
setup_logging(level=os.getenv("LOG_LEVEL", "WARNING").upper(), log_file=os.getenv("LOG_FILE", None))
