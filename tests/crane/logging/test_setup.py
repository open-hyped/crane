import logging

from crane.logging.setup import setup_logging


class TestSetupLogging:
    def test_it_leaves_an_application_s_own_loggers_alone(self):
        # `dictConfig` disables existing loggers by default, and this runs on import: a
        # user who configured logging before importing crane would lose all of it.
        mine = logging.getLogger("some_application_that_was_here_first")
        assert not mine.disabled

        setup_logging("INFO")

        assert not mine.disabled
