import json
import os
from dataclasses import dataclass
from typing import ClassVar

import datasets
import pytest

from crane import ArrowDatasetWriter, JsonDatasetWriter
from crane.distributed.core._entrypoint import run_finalize, run_worker
from crane.distributed.core.base import BackendView, DistributedBackend, RunState
from crane.distributed.core.job import get_job_info, reset_job_info

NUM_SHARDS, ROWS_PER_SHARD = 6, 5


@dataclass(frozen=True)
class _LocalBackend(DistributedBackend):
    """Submits nothing. Lets the jobs be run in-process, as the cluster would run them."""

    name: ClassVar[str] = "local-for-tests"

    num_jobs: int = 3
    work_dir: None | str = None
    cpus: int = 1

    @property
    def cpus_per_job(self) -> None | int:
        # never the machine's core count: a test should not spawn one worker per core
        return self.cpus

    def submit(self, spec):
        return ["local"]

    def poll(self, spec, job_ids):
        return BackendView(active_jobs={}, finalize_pending=False)

    def cancel(self, spec, job_ids):
        pass


@pytest.fixture
def ds() -> datasets.IterableDataset:
    # Arrow-backed, as a loaded dataset is. Not `from_generator`: that produces an examples
    # iterable with no arrow path, which the multiprocessing runner cannot shard - nothing
    # to do with distribution, but it would fail every test below that uses more than one
    # process.
    rows = [
        {"shard": shard, "value": shard * ROWS_PER_SHARD + i}
        for shard in range(NUM_SHARDS)
        for i in range(ROWS_PER_SHARD)
    ]
    ds = datasets.Dataset.from_list(rows).to_iterable_dataset(num_shards=NUM_SHARDS)
    # a lambda, so the payload exercises the dill path a real submission depends on
    return ds.map(
        lambda x: {"doubled": x["value"] * 2},
        features=datasets.Features(dict(ds.features) | {"doubled": datasets.Value("int64")}),
    )


@pytest.fixture(autouse=True)
def isolate_jobs():
    """Give every job a process that is not already part of a run.

    On a cluster each job has one to itself; here they share the test process, so the
    identity a job establishes has to be cleared the way exiting would clear it.
    """
    reset_job_info()
    yield
    reset_job_info()


def _tag_with_job_index(row: dict) -> dict:
    """Stand in for user code that needs to know where it is running."""
    info = get_job_info()
    return {"job": -1 if info is None else info.index}


def _run_all_jobs(run) -> None:
    """Do what the cluster would do: every worker, then the finalize job."""
    for job_index in range(run.num_jobs):
        assert run_worker(run.run_dir, job_index) == 0, f"job {job_index} failed"
    assert run_finalize(run.run_dir) == 0


class TestDistributedWrite:
    def test_every_row_is_written_exactly_once(self, ds, tmp_path):
        out = str(tmp_path / "out")
        run = ArrowDatasetWriter(out, disable_tqdm=True).submit(ds, on=_LocalBackend(num_jobs=3))
        _run_all_jobs(run)

        loaded = datasets.load_from_disk(out)
        assert sorted(loaded["value"]) == list(range(NUM_SHARDS * ROWS_PER_SHARD))

    def test_the_transform_ran_in_the_jobs(self, ds, tmp_path):
        out = str(tmp_path / "out")
        run = ArrowDatasetWriter(out, disable_tqdm=True).submit(ds, on=_LocalBackend(num_jobs=3))
        _run_all_jobs(run)

        loaded = datasets.load_from_disk(out)
        assert all(row["doubled"] == row["value"] * 2 for row in loaded)

    def test_a_transform_can_ask_which_job_it_is_running_in(self, ds, tmp_path):
        # A transform is handed a row and nothing else, so asking is the only way it can
        # know. Two processes, so the answer has to survive the job starting a worker;
        # that it survives `spawn` as well as `fork` is `test_job.py`'s to show.
        out = str(tmp_path / "out")
        tagged = ds.map(
            _tag_with_job_index,
            features=datasets.Features(dict(ds.features) | {"job": datasets.Value("int64")}),
        )
        run = ArrowDatasetWriter(out, disable_tqdm=True).submit(
            tagged, on=_LocalBackend(num_jobs=3, cpus=2)
        )
        _run_all_jobs(run)

        loaded = datasets.load_from_disk(out)
        assert set(loaded["job"]) == {0, 1, 2}

    def test_a_local_write_is_not_part_of_a_run(self, ds, tmp_path):
        # the same transform, written locally: nothing set it, so there is no job to name
        out = str(tmp_path / "out")
        tagged = ds.map(
            _tag_with_job_index,
            features=datasets.Features(dict(ds.features) | {"job": datasets.Value("int64")}),
        )
        ArrowDatasetWriter(out, disable_tqdm=True).write(tagged)

        assert set(datasets.load_from_disk(out)["job"]) == {-1}

    def test_shard_names_carry_the_job_index_and_do_not_collide(self, ds, tmp_path):
        out = str(tmp_path / "out")
        run = ArrowDatasetWriter(out, disable_tqdm=True).submit(ds, on=_LocalBackend(num_jobs=3))
        _run_all_jobs(run)

        shards = sorted(f for f in os.listdir(out) if f.startswith("shard-"))
        assert len(shards) == len(set(shards))
        assert all(f.startswith(("shard-000-", "shard-001-", "shard-002-")) for f in shards)

    def test_the_run_directory_is_not_listed_as_data(self, ds, tmp_path):
        # `.crane` lives inside the save directory; listing it would break load_from_disk
        out = str(tmp_path / "out")
        run = ArrowDatasetWriter(out, disable_tqdm=True).submit(ds, on=_LocalBackend(num_jobs=3))
        _run_all_jobs(run)

        with open(os.path.join(out, "state.json")) as f:
            names = [entry["filename"] for entry in json.load(f)["_data_files"]]

        assert ".crane" not in names
        assert all(name.startswith("shard-") for name in names)

    def test_each_job_reports_and_marks_its_shards(self, ds, tmp_path):
        out = str(tmp_path / "out")
        run = ArrowDatasetWriter(out, disable_tqdm=True).submit(ds, on=_LocalBackend(num_jobs=3))
        _run_all_jobs(run)

        assert sorted(os.listdir(os.path.join(run.run_dir, "jobs"))) == [
            "0.json",
            "1.json",
            "2.json",
            # the finalize job reports too, so a run whose metadata was never written is
            # not mistaken for one that completed
            "finalize.json",
        ]
        # one marker per shard, named per job so two jobs cannot overwrite each other
        assert len(os.listdir(os.path.join(run.run_dir, "shards"))) == NUM_SHARDS

    def test_the_run_reports_itself_completed(self, ds, tmp_path):
        out = str(tmp_path / "out")
        run = ArrowDatasetWriter(out, disable_tqdm=True).submit(ds, on=_LocalBackend(num_jobs=3))
        _run_all_jobs(run)

        status = run.status()
        assert status.state is RunState.COMPLETED
        assert status.shards_completed == NUM_SHARDS
        run.wait(poll_interval=0.01)

    def test_matches_what_a_local_write_produces(self, ds, tmp_path):
        distributed, local = str(tmp_path / "dist"), str(tmp_path / "local")

        run = ArrowDatasetWriter(distributed, disable_tqdm=True).submit(
            ds, on=_LocalBackend(num_jobs=3)
        )
        _run_all_jobs(run)
        ArrowDatasetWriter(local, num_proc=1, disable_tqdm=True).write(ds)

        # the point of the feature: only the machinery differs, never the result
        assert sorted(datasets.load_from_disk(distributed)["value"]) == sorted(
            datasets.load_from_disk(local)["value"]
        )

    def test_more_jobs_than_shards_is_clamped(self, ds, tmp_path):
        out = str(tmp_path / "out")
        run = ArrowDatasetWriter(out, disable_tqdm=True).submit(ds, on=_LocalBackend(num_jobs=99))
        assert run.num_jobs == NUM_SHARDS

        _run_all_jobs(run)
        assert sorted(datasets.load_from_disk(out)["value"]) == list(
            range(NUM_SHARDS * ROWS_PER_SHARD)
        )

    def test_jobs_use_the_multiprocessing_runner(self, ds, tmp_path):
        # A distributed job defaults `num_proc` to the cores it reserved, so the runner a
        # real job uses is the multiprocessing one. Every other test here would pass with
        # that path completely broken.
        out = str(tmp_path / "out")
        run = ArrowDatasetWriter(out, disable_tqdm=True).submit(
            ds, on=_LocalBackend(num_jobs=2, cpus=2)
        )
        _run_all_jobs(run)

        loaded = datasets.load_from_disk(out)
        assert sorted(loaded["value"]) == list(range(NUM_SHARDS * ROWS_PER_SHARD))

    def test_another_writer_needs_no_changes(self, ds, tmp_path):
        # the shard naming lives in the base class, so every writer gets it
        out = str(tmp_path / "out")
        run = JsonDatasetWriter(out, disable_tqdm=True).submit(ds, on=_LocalBackend(num_jobs=2))
        _run_all_jobs(run)

        shards = sorted(f for f in os.listdir(out) if f.startswith("shard-"))
        assert all(f.endswith(".json") for f in shards)
        assert len(shards) == len(set(shards))


class TestFailingWorkload:
    def test_a_raising_transform_is_reported_not_swallowed(self, tmp_path):
        def boom(x):
            raise ValueError("nope")

        ds = datasets.Dataset.from_list([{"a": 0}, {"a": 1}]).to_iterable_dataset(num_shards=2)
        ds = ds.map(boom, features=ds.features)

        out = str(tmp_path / "out")
        run = ArrowDatasetWriter(out, disable_tqdm=True).submit(ds, on=_LocalBackend(num_jobs=2))

        assert run_worker(run.run_dir, 0) == 1

        with open(run._spec.job_result_path(0)) as f:
            failures = json.load(f)["failures"]

        assert failures and failures[0]["error_type"] == "ValueError"
        assert "nope" in failures[0]["error_message"]

    def test_a_job_that_fails_before_starting_reports_rather_than_raising(self, ds, tmp_path):
        # Everything a job does, setting up included, has to end in a report: a job that
        # raised its way out would leave the run's handle waiting on a job it cannot see.
        out = str(tmp_path / "out")
        run = ArrowDatasetWriter(out, disable_tqdm=True).submit(ds, on=_LocalBackend(num_jobs=2))

        assert run_worker(run.run_dir, 7) == 1

        with open(run._spec.job_result_path(7)) as f:
            result = json.load(f)

        assert "out of range" in result["error"]
