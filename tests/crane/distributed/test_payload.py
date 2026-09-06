import socket
import sqlite3
from dataclasses import dataclass
from typing import ClassVar

import datasets
import pytest

from crane import ArrowDatasetWriter
from crane.distributed.core import payload
from crane.distributed.core.base import BackendView, DistributedBackend


@dataclass(frozen=True)
class _CountingBackend(DistributedBackend):
    """A backend that queues nothing, so a submission can be measured."""

    name: ClassVar[str] = "counting-for-tests"

    num_jobs: int = 1
    work_dir: None | str = None

    @property
    def cpus_per_job(self) -> None | int:
        return 1

    def submit(self, spec):
        return ["1"]

    def poll(self, spec, job_ids):
        return BackendView(active_jobs={}, finalize_pending=False)

    def cancel(self, spec, job_ids):
        pass


@pytest.fixture
def ds() -> datasets.IterableDataset:
    return datasets.Dataset.from_list([{"a": i} for i in range(4)]).to_iterable_dataset()


class TestRoundTrip:
    def test_payload_survives_the_trip(self, ds, tmp_path):
        writer = ArrowDatasetWriter(str(tmp_path / "out"), num_proc=2)
        path = str(tmp_path / "payload.dill")

        payload.dump(payload.Payload(ds=ds, target=writer, num_jobs=4), path)
        loaded = payload.load(path)

        assert loaded.num_jobs == 4
        assert loaded.target.save_dir == writer.save_dir
        assert loaded.target.num_proc == 2
        assert [s["a"] for s in loaded.ds] == [0, 1, 2, 3]

    def test_a_lambda_transform_survives(self, ds, tmp_path):
        # the reason this is dill and not pickle
        ds = ds.map(lambda x: {"b": x["a"] * 2})
        path = str(tmp_path / "payload.dill")

        payload.dump(payload.Payload(ds=ds, target=None, num_jobs=1), path)
        assert [s["b"] for s in payload.load(path).ds] == [0, 2, 4, 6]

    def test_a_closure_finalizer_survives(self, ds, tmp_path):
        factor = 3
        path = str(tmp_path / "payload.dill")

        payload.dump(
            payload.Payload(ds=ds, target=None, num_jobs=1, finalizer=lambda b: b["a"] * factor),
            path,
        )
        assert payload.load(path).finalizer({"a": 2}) == 6


class TestCheck:
    def test_accepts_what_can_travel(self, ds):
        payload.check(payload.Payload(ds=ds, target=None, num_jobs=1))

    def test_rejects_process_local_state_with_a_useful_error(self, ds):
        # a database connection is the realistic version of "closes over something that
        # only exists in this process"
        conn = sqlite3.connect(":memory:")
        try:
            data = payload.Payload(
                ds=ds, target=None, num_jobs=1, finalizer=lambda b: conn.execute("select 1")
            )
            with pytest.raises(TypeError, match="cannot be sent to a distributed job"):
                payload.check(data)
        finally:
            conn.close()

    def test_rejects_a_socket(self, ds):
        sock = socket.socket()
        try:
            data = payload.Payload(ds=ds, target=None, num_jobs=1, finalizer=lambda b: sock)
            with pytest.raises(TypeError, match="database connection, a socket"):
                payload.check(data)
        finally:
            sock.close()

    def test_an_open_file_is_not_caught(self, ds, tmp_path):
        # Pins a known gap rather than asserting the behaviour is right: dill takes a file
        # handle by path and reopens it inside the job, so this passes the check and is
        # still the wrong thing to close over. If dill ever starts refusing it, this test
        # fails and the docstring on `check` needs updating with it.
        handle = open(tmp_path / "scratch", "w")
        try:
            data = payload.Payload(
                ds=ds, target=None, num_jobs=1, finalizer=lambda b: handle.write("x")
            )
            payload.check(data)
        finally:
            handle.close()


class TestSerializedOnce:
    """Serializing is the expensive half of submitting, so it has to happen once.

    A payload carrying a tokenizer or a model is a large object graph and `recurse=True`
    walks all of it: a real submission measured ~24 minutes per pass. Checking and writing
    used to take a pass each, which is invisible against the toy payloads above and costs
    half an hour against a real one.
    """

    def test_writing_a_payload_serializes_it_once(self, ds, tmp_path, monkeypatch):
        calls = []
        real_dumps = payload.dill.dumps

        def counting_dumps(*args, **kwargs):
            calls.append(1)
            return real_dumps(*args, **kwargs)

        monkeypatch.setattr(payload.dill, "dumps", counting_dumps)
        data = payload.Payload(ds=ds, target=None, num_jobs=1)

        payload.dump(data, str(tmp_path / "payload.dill"))

        assert len(calls) == 1

    def test_the_written_payload_is_still_checked(self, ds, tmp_path):
        # the pass that writes is also the pass that verifies, so an unsendable payload
        # still fails at submission rather than inside a job
        conn = sqlite3.connect(":memory:")
        try:
            data = payload.Payload(ds=ds, target=None, num_jobs=1, finalizer=lambda x: conn)
            with pytest.raises(TypeError, match="cannot be sent to a distributed job"):
                payload.dump(data, str(tmp_path / "payload.dill"))
        finally:
            conn.close()

    def test_a_submission_serializes_the_payload_once(self, ds, tmp_path, monkeypatch):
        # the end the cost is actually paid at
        from crane.distributed.core import run as run_module

        calls = []
        real_dumps = payload.dill.dumps

        def counting_dumps(*args, **kwargs):
            calls.append(1)
            return real_dumps(*args, **kwargs)

        monkeypatch.setattr(payload.dill, "dumps", counting_dumps)
        writer = ArrowDatasetWriter(str(tmp_path / "out"), num_proc=1)

        run_module.submit(
            ds=ds,
            target=writer,
            backend=_CountingBackend(work_dir=str(tmp_path / "work")),
            save_dir=str(tmp_path / "out"),
            needs_finalize=True,
        )

        assert len(calls) == 1
