import json
import os
from dataclasses import dataclass, field, replace
from typing import ClassVar

import pytest

from crane.core.runners.base import ShardProcessingError
from crane.distributed.core.base import BackendView, DistributedBackend, JobState, RunSpec, RunState
from crane.distributed.core.run import DistributedRun, JobLostError, RunNotFoundError, attach


@dataclass(frozen=True)
class _FakeBackend(DistributedBackend):
    """A backend that reports whatever a test tells it to."""

    name: ClassVar[str] = "fake-for-tests"

    num_jobs: int = 2
    active: dict = field(default_factory=dict)
    finalize_pending: bool = False
    cancelled: list = field(default_factory=list)

    def submit(self, spec):
        return ["1"]

    def poll(self, spec, job_ids):
        return BackendView(
            active_jobs={int(k): JobState(v) for k, v in self.active.items()},
            finalize_pending=self.finalize_pending,
        )

    def cancel(self, spec, job_ids):
        self.cancelled.append(list(job_ids))
        # a cancelled job leaves the queue, which is what lets `wait` finish
        self.active.clear()


def _failure(rank: int, shard_id: int) -> dict:
    return {
        "rank": rank,
        "shard_id": shard_id,
        "error_type": "ValueError",
        "error_message": "boom",
        "stack_trace": "Traceback...\nValueError: boom\n",
    }


@pytest.fixture
def spec(tmp_path) -> RunSpec:
    run_dir = tmp_path / "run"
    (run_dir / "jobs").mkdir(parents=True)
    (run_dir / "shards").mkdir(parents=True)
    spec = RunSpec(
        run_id="out-abc123",
        run_dir=str(run_dir),
        num_jobs=2,
        num_shards=8,
        save_dir=str(tmp_path / "out"),
        python="/usr/bin/python",
        needs_finalize=True,
        backend={"name": "fake-for-tests"},
        failure_policy="fail_fast",
        job_ids=["1"],
    )
    spec.save()
    return spec


def _report(spec: RunSpec, job_index: int, failures: list = []) -> None:
    with open(spec.job_result_path(job_index), "w") as f:
        json.dump({"job_index": job_index, "failures": failures, "error": None}, f)


def _mark_shards(spec: RunSpec, job_index: int, count: int) -> None:
    for shard_id in range(count):
        with open(spec.shard_marker_path(job_index, shard_id), "w"):
            pass


def _run(spec: RunSpec, **backend_kwargs) -> DistributedRun:
    return DistributedRun(spec, _FakeBackend(**backend_kwargs))


class TestStatus:
    def test_pending_before_anything_starts(self, spec):
        assert _run(spec, active={0: "pending", 1: "pending"}).status().state is RunState.PENDING

    def test_running_while_any_job_is_alive(self, spec):
        _report(spec, 0)
        status = _run(spec, active={1: "running"}).status()
        assert status.state is RunState.RUNNING
        assert status.jobs == {0: JobState.COMPLETED, 1: JobState.RUNNING}

    def test_finalizing_once_the_workers_are_done(self, spec):
        _report(spec, 0)
        _report(spec, 1)
        assert _run(spec, finalize_pending=True).status().state is RunState.FINALIZING

    def test_completed_once_the_metadata_is_written(self, spec):
        _report(spec, 0)
        _report(spec, 1)
        assert _run(spec).status().state is RunState.COMPLETED

    def test_counts_shards_across_every_job(self, spec):
        _mark_shards(spec, 0, 3)
        _mark_shards(spec, 1, 2)
        run = _run(spec, active={0: "running", 1: "running"})
        assert run.status().shards_completed == 5

    def test_markers_of_different_jobs_do_not_collide(self, spec):
        # both jobs number their shards from zero; the marker name carries the job index
        _mark_shards(spec, 0, 2)
        _mark_shards(spec, 1, 2)
        assert _run(spec, active={0: "running"}).status().shards_completed == 4

    def test_a_job_that_reported_failures_is_failed(self, spec):
        _report(spec, 0, failures=[_failure(0, 2)])
        _report(spec, 1)
        status = _run(spec).status()
        assert status.state is RunState.FAILED
        assert status.jobs[0] is JobState.FAILED
        assert [f.shard_id for f in status.failures] == [2]

    def test_a_job_that_vanished_is_lost(self, spec):
        # gone from the scheduler and never wrote a result
        _report(spec, 0)
        status = _run(spec).status()
        assert status.jobs[1] is JobState.LOST
        assert status.lost_jobs == [1]
        assert status.state is RunState.FAILED

    def test_failure_wins_over_a_queued_finalize_job(self, spec):
        # the finalize job depends on afterok and will never start
        _report(spec, 0, failures=[_failure(0, 1)])
        _report(spec, 1)
        assert _run(spec, finalize_pending=True).status().state is RunState.FAILED


class TestWait:
    def test_returns_quietly_on_success(self, spec):
        _report(spec, 0)
        _report(spec, 1)
        _run(spec).wait(poll_interval=0.01)

    def test_raises_the_same_error_a_local_run_raises(self, spec):
        _report(spec, 0, failures=[_failure(0, 2)])
        _report(spec, 1, failures=[_failure(1, 5)])

        with pytest.raises(ShardProcessingError) as info:
            _run(spec).wait(poll_interval=0.01)

        # every failure from every job, not just the first
        assert sorted(f.shard_id for f in info.value.failures) == [2, 5]

    def test_a_lost_job_is_not_dressed_up_as_a_workload_failure(self, spec):
        _report(spec, 0)
        with pytest.raises(JobLostError, match="outside it"):
            _run(spec).wait(poll_interval=0.01)

    def test_times_out_without_cancelling(self, spec):
        run = _run(spec, active={0: "running", 1: "running"})
        with pytest.raises(TimeoutError, match="has not been cancelled"):
            run.wait(timeout=0.05, poll_interval=0.01)


class TestStopTheRestOnFailure:
    """Under fail-fast the handle stops the jobs that are still running.

    Locally the runner stops its own workers on the first failure. The jobs of a distributed
    run have no connection to each other, so nothing does it unless the handle does.
    """

    def test_a_failure_stops_the_jobs_still_running(self, spec):
        _report(spec, 0, failures=[_failure(0, 2)])
        run = _run(spec, active={1: "running"})

        # bounded, so that a run which is never stopped fails the test rather than
        # hanging the suite; stopping it correctly returns in milliseconds
        with pytest.raises(ShardProcessingError):
            run.wait(timeout=5, poll_interval=0.01)

        assert run._backend.cancelled == [["1"]], "the remaining jobs were left running"

    def test_the_real_failure_is_reported_not_the_jobs_we_killed(self, spec):
        # cancelling makes the siblings look lost; the workload failure is the fault
        _report(spec, 0, failures=[_failure(0, 2)])
        run = _run(spec, active={1: "running"})

        with pytest.raises(ShardProcessingError) as info:
            run.wait(timeout=5, poll_interval=0.01)

        assert [f.shard_id for f in info.value.failures] == [2]

    def test_the_run_is_stopped_only_once(self, spec):
        _report(spec, 0, failures=[_failure(0, 2)])
        run = _run(spec, active={1: "running"})

        with pytest.raises(ShardProcessingError):
            run.wait(timeout=5, poll_interval=0.01)

        assert len(run._backend.cancelled) == 1

    def test_skip_shard_lets_the_other_jobs_carry_on(self, spec):
        spec = RunSpec(**(spec.__dict__ | {"failure_policy": "skip_shard"}))
        spec.save()
        _report(spec, 0, failures=[_failure(0, 2)])
        _report(spec, 1)
        run = _run(spec)

        with pytest.raises(ShardProcessingError):
            run.wait(poll_interval=0.01)

        assert run._backend.cancelled == [], "a skipped shard should not stop the run"

    def test_a_lost_job_alone_does_not_stop_the_run(self, spec):
        # a preemption is not the workload failing, and the rest may still be worth finishing
        run = _run(spec, active={1: "running"})
        assert run.status().lost_jobs == [0], "the test needs a job that vanished"

        with pytest.raises(TimeoutError):
            run.wait(timeout=0.05, poll_interval=0.01)

        assert run._backend.cancelled == []

    def test_watching_stops_the_rest_just_as_waiting_does(self, spec):
        # following a run means the same thing whichever method does it
        _report(spec, 0, failures=[_failure(0, 2)])
        run = _run(spec, active={1: "running"})

        run.watch(timeout=5, poll_interval=0.01)

        assert run._backend.cancelled == [["1"]], "watch left the remaining jobs running"

    def test_watching_a_skipped_shard_lets_the_run_carry_on(self, spec):
        spec = RunSpec(**(spec.__dict__ | {"failure_policy": "skip_shard"}))
        spec.save()
        _report(spec, 0, failures=[_failure(0, 2)])
        _report(spec, 1)
        run = _run(spec)

        run.watch(timeout=5, poll_interval=0.01)

        assert run._backend.cancelled == []


class TestLogs:
    def test_reads_a_job_log(self, spec):
        logs = os.path.join(spec.run_dir, "logs")
        os.makedirs(logs)
        with open(os.path.join(logs, "500_1.out"), "w") as f:
            f.write("hello from job 1")

        assert _run(spec).logs(1) == "hello from job 1"

    def test_says_so_when_there_is_no_log(self, spec):
        assert "No out log found" in _run(spec).logs(1)


class TestAttach:
    def test_rebuilds_the_handle_from_the_run_directory(self, spec):
        run = attach(spec.run_dir)
        assert run.run_id == spec.run_id
        assert run.num_jobs == spec.num_jobs
        assert run.job_ids == ["1"]

    def test_rejects_a_directory_that_is_not_a_run(self, tmp_path):
        with pytest.raises(RunNotFoundError, match="does not look like"):
            attach(str(tmp_path))

    def test_attached_handle_reads_the_same_state(self, spec):
        _report(spec, 0)
        _report(spec, 1)
        assert attach(spec.run_dir).status().state is RunState.COMPLETED


class TestWorkDir:
    def test_defaults_beside_the_dataset(self, tmp_path):
        from crane.distributed.core.run import _work_dir

        assert _work_dir(_FakeBackend(), "/data/out") == os.path.join("/data/out", ".crane")

    def test_a_consumer_without_a_work_dir_is_an_error(self):
        from crane.distributed.core.run import _work_dir

        with pytest.raises(ValueError, match="Pass `work_dir`"):
            _work_dir(_FakeBackend(), None)


class TestRunId:
    def test_names_the_run_after_the_output(self):
        from crane.distributed.core.run import _make_run_id

        assert _make_run_id(None, "/shared/fineweb-annotated").startswith("fineweb-annotated-")

    def test_is_unique_per_submission(self):
        from crane.distributed.core.run import _make_run_id

        # a resubmission must not collide with the run it repeats
        assert _make_run_id(None, "/out") != _make_run_id(None, "/out")
