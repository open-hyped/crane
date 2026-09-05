import socket
import sqlite3

import datasets
import pytest

from crane import ArrowDatasetWriter
from crane.distributed.core import payload


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
