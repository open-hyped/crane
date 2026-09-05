import json
import os
from dataclasses import dataclass
from typing import ClassVar

import datasets
import pytest

from crane import ArrowDatasetWriter, JsonDatasetWriter
from crane.dist.core._entrypoint import run_finalize, run_worker
from crane.dist.core.base import BackendView, DistributedBackend, RunState

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
            "0.json", "1.json", "2.json"
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
