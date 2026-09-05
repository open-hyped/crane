import multiprocessing as mp
import os

import pytest

from crane.distributed.core.job import (
    INDEX_ENV_VAR,
    NUM_JOBS_ENV_VAR,
    RUN_ID_ENV_VAR,
    JobInfo,
    get_job_info,
    reset_job_info,
    set_job_info,
)


@pytest.fixture(autouse=True)
def clean_job_info():
    """Job info is process state, so a test that set it would leak into the next."""
    reset_job_info()
    yield
    reset_job_info()


def _read_in_child(queue):
    queue.put(get_job_info())


class TestJobInfo:
    def test_a_local_process_is_not_part_of_a_run(self):
        assert get_job_info() is None

    def test_the_entrypoint_sets_it(self):
        info = set_job_info(index=3, num_jobs=8, run_id="out-1")

        assert info == JobInfo(index=3, num_jobs=8, run_id="out-1")
        assert get_job_info() == info

    def test_it_cannot_be_set_twice(self):
        set_job_info(index=0, num_jobs=1, run_id="out-1")

        with pytest.raises(AssertionError, match="already set"):
            set_job_info(index=0, num_jobs=1, run_id="out-1")

    @pytest.mark.parametrize("index", [-1, 4, 100])
    def test_an_index_outside_the_run_is_rejected(self, index):
        with pytest.raises(ValueError, match="out of range"):
            set_job_info(index=index, num_jobs=4, run_id="out-1")


class TestPropagation:
    def test_setting_it_exports_it(self):
        set_job_info(index=3, num_jobs=8, run_id="out-1")

        assert os.environ[INDEX_ENV_VAR] == "3"
        assert os.environ[NUM_JOBS_ENV_VAR] == "8"
        assert os.environ[RUN_ID_ENV_VAR] == "out-1"

    def test_the_environment_answers_a_process_that_never_set_it(self, monkeypatch):
        # the case of a worker the job started: the global is gone, the environment is not
        monkeypatch.setenv(INDEX_ENV_VAR, "3")
        monkeypatch.setenv(NUM_JOBS_ENV_VAR, "8")
        monkeypatch.setenv(RUN_ID_ENV_VAR, "out-1")

        assert get_job_info() == JobInfo(index=3, num_jobs=8, run_id="out-1")

    @pytest.mark.parametrize("missing", [INDEX_ENV_VAR, NUM_JOBS_ENV_VAR, RUN_ID_ENV_VAR])
    def test_a_partial_environment_is_not_a_run(self, monkeypatch, missing):
        for var, value in ((INDEX_ENV_VAR, "3"), (NUM_JOBS_ENV_VAR, "8"), (RUN_ID_ENV_VAR, "o")):
            if var != missing:
                monkeypatch.setenv(var, value)

        assert get_job_info() is None

    def test_a_nonsense_environment_is_ignored_rather_than_raising(self, monkeypatch):
        # only reachable if something other than crane wrote these; failing here would fail
        # at whatever depth the caller happened to ask from
        monkeypatch.setenv(INDEX_ENV_VAR, "not-a-number")
        monkeypatch.setenv(NUM_JOBS_ENV_VAR, "8")
        monkeypatch.setenv(RUN_ID_ENV_VAR, "out-1")

        with pytest.warns(UserWarning, match="not both integers"):
            assert get_job_info() is None

    @pytest.mark.parametrize("start_method", ["fork", "spawn"])
    def test_a_child_process_inherits_it(self, start_method):
        # the point of going through the environment: `crane.core`'s runners know nothing
        # about distribution, so they cannot carry this across for us
        set_job_info(index=3, num_jobs=8, run_id="out-1")

        ctx = mp.get_context(start_method)
        queue = ctx.Queue()
        proc = ctx.Process(target=_read_in_child, args=(queue,))
        proc.start()
        try:
            assert queue.get(timeout=60) == JobInfo(index=3, num_jobs=8, run_id="out-1")
        finally:
            proc.join(timeout=60)

    def test_resetting_it_clears_the_environment_too(self):
        set_job_info(index=3, num_jobs=8, run_id="out-1")
        reset_job_info()

        assert get_job_info() is None
        assert INDEX_ENV_VAR not in os.environ


class TestNestedRuns:
    def test_a_job_overrides_an_identity_it_inherited(self, monkeypatch):
        # a run submitted from inside another job inherits that job's identity; the job
        # that reads it is the one that has to win
        monkeypatch.setenv(INDEX_ENV_VAR, "3")
        monkeypatch.setenv(NUM_JOBS_ENV_VAR, "8")
        monkeypatch.setenv(RUN_ID_ENV_VAR, "outer")
        assert get_job_info().run_id == "outer"

        reset_job_info()
        info = set_job_info(index=0, num_jobs=2, run_id="inner")

        assert info == JobInfo(index=0, num_jobs=2, run_id="inner")
        assert os.environ[RUN_ID_ENV_VAR] == "inner"
