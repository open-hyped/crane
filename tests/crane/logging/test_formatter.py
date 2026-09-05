import logging

import pytest

from crane.core.worker import reset_worker_info, set_worker_info
from crane.distributed.core.job import reset_job_info, set_job_info
from crane.logging.formatter import RankAwareFormatter


@pytest.fixture(autouse=True)
def clean_process_info():
    """Both are process state, so a test that set one would leak into the next."""
    reset_worker_info()
    reset_job_info()
    yield
    reset_worker_info()
    reset_job_info()


def _prefix() -> str:
    record = logging.LogRecord("crane.test", logging.INFO, __file__, 1, "hello", None, None)
    RankAwareFormatter("%(rank_prefix)s%(message)s").format(record)
    return record.rank_prefix


class TestRankAwareFormatter:
    def test_a_plain_local_run_is_not_prefixed(self):
        assert _prefix() == ""

    def test_a_local_worker_is_named_by_its_rank(self):
        # counted from one: the third of four workers, not "rank 2"
        set_worker_info(rank=2, num_workers=4, seed=None)

        assert _prefix() == "[Rank 3/4] "

    def test_a_job_is_named_before_it_starts_any_worker(self):
        set_job_info(index=3, num_jobs=8, run_id="out-1")

        assert _prefix() == "[Job 4/8] "

    def test_a_worker_of_a_job_is_named_by_both(self):
        # every job numbers its workers from zero, so the rank alone does not say which of
        # a run's log files a line came from
        set_job_info(index=3, num_jobs=8, run_id="out-1")
        set_worker_info(rank=2, num_workers=4, seed=None)

        assert _prefix() == "[Job 4/8 Rank 3/4] "

    def test_the_prefix_reaches_the_message(self):
        set_job_info(index=3, num_jobs=8, run_id="out-1")
        record = logging.LogRecord("crane.test", logging.INFO, __file__, 1, "hello", None, None)

        assert RankAwareFormatter("%(rank_prefix)s%(message)s").format(record) == "[Job 4/8] hello"


class TestNoRecursion:
    def test_a_malformed_environment_does_not_send_the_formatter_round_in_circles(
        self, monkeypatch
    ):
        # The formatter asks which job this is for every record, so anything that question
        # logs on its way to an answer would be formatted by asking it again.
        monkeypatch.setenv("CRANE_DISTRIBUTED_INDEX", "not-a-number")
        monkeypatch.setenv("CRANE_DISTRIBUTED_NUM_JOBS", "8")
        monkeypatch.setenv("CRANE_DISTRIBUTED_RUN_ID", "out-1")

        with pytest.warns(UserWarning, match="not both integers"):
            assert _prefix() == ""


class TestCounting:
    def test_the_last_job_and_worker_read_as_the_total(self):
        # the point of counting from one: the end of a run looks like the end of a run
        set_job_info(index=7, num_jobs=8, run_id="out-1")
        set_worker_info(rank=15, num_workers=16, seed=None)

        assert _prefix() == "[Job 8/8 Rank 16/16] "

    def test_a_single_job_of_a_single_worker_still_says_so(self):
        set_job_info(index=0, num_jobs=1, run_id="out-1")
        set_worker_info(rank=0, num_workers=1, seed=None)

        assert _prefix() == "[Job 1/1 Rank 1/1] "
