import json
import os
from dataclasses import dataclass
from typing import ClassVar

import pytest

from crane.dist.core.base import (
    BackendView,
    DistributedBackend,
    JobState,
    RunSpec,
    RunState,
    RunStatus,
)
from crane.dist.slurm import Slurm


@dataclass(frozen=True)
class _StubBackend(DistributedBackend):
    name: ClassVar[str] = "stub-for-tests"

    num_jobs: int = 1

    def submit(self, spec):
        return ["1"]

    def poll(self, spec, job_ids):
        return BackendView(active_jobs={}, finalize_pending=False)

    def cancel(self, spec, job_ids):
        pass


@pytest.fixture
def spec(tmp_path) -> RunSpec:
    return RunSpec(
        run_id="out-abc123",
        run_dir=str(tmp_path),
        num_jobs=4,
        num_shards=16,
        save_dir="/data/out",
        python="/usr/bin/python",
        needs_finalize=True,
        backend={"name": "stub-for-tests", "num_jobs": 4},
    )


class TestRunSpec:
    def test_round_trips_through_the_run_directory(self, spec):
        spec.save()
        assert RunSpec.load(spec.run_dir) == spec

    def test_is_plain_json(self, spec):
        spec.save()
        with open(os.path.join(spec.run_dir, "spec.json")) as f:
            assert json.load(f)["run_id"] == "out-abc123"

    def test_marker_path_is_unique_per_job(self, spec):
        # shard ids only count within a job, so the job index has to be in the name
        assert spec.shard_marker_path(0, 3) != spec.shard_marker_path(1, 3)


class TestBackendRegistry:
    def test_backend_round_trips_through_a_spec(self):
        backend = Slurm(num_jobs=8, partition="batch", cpus_per_task=4, env={"A": "b"})
        assert DistributedBackend.from_dict(backend.to_dict()) == backend

    def test_unknown_backend_is_named_in_the_error(self):
        with pytest.raises(ValueError, match="not available here"):
            DistributedBackend.from_dict({"name": "nonexistent"})

    def test_subclass_registers_itself(self):
        assert DistributedBackend.from_dict({"name": "stub-for-tests", "num_jobs": 2}).num_jobs == 2


class TestRunStatus:
    def test_summarises_itself(self):
        status = RunStatus(
            state=RunState.RUNNING,
            jobs={0: JobState.COMPLETED, 1: JobState.RUNNING},
            shards_completed=5,
            num_shards=10,
        )
        assert "5/10 shards" in str(status)
        assert "1/2 jobs done" in str(status)
