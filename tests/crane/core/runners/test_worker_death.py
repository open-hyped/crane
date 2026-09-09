"""A worker killed outright must fail the run, not hang it.

A worker that raises reports the traceback and exits tidily. This covers the other kind of
death - the process taken away with no chance to say anything, as an out-of-memory kill
does. Nothing else in the run can see it: a worker reports DONE from a `finally`, and the
message loop counts those to know when the run is over, so a killed worker leaves the loop
waiting for a message that can never arrive.

Before this was handled, a real run sat for three hours after its last byte was written.
"""

import os
import signal
import time

import datasets
import pytest

from crane import ArrowDatasetWriter
from crane.core.consumer import DatasetConsumer
from crane.core.runners.base import WorkerDiedError

# every process the runner starts, so a test can kill one
_CHILDREN_BEFORE: list[int] = []


def _suicide_on_one_shard(example, target_shard: int):
    """Kill this process mid-run, the way an out-of-memory kill would.

    SIGKILL rather than an exception on purpose: an exception is reported and handled, and
    this is about the death that reports nothing.
    """
    if example["shard"] == target_shard:
        os.kill(os.getpid(), signal.SIGKILL)
    # the others dawdle, so the run is still going when the death happens
    time.sleep(0.05)
    return example


@pytest.fixture
def ds() -> datasets.IterableDataset:
    rows = [{"shard": s, "value": s * 100 + i} for s in range(4) for i in range(100)]
    return datasets.Dataset.from_list(rows).to_iterable_dataset(num_shards=4)


class TestWorkerDeath:
    def test_a_killed_worker_fails_the_run(self, ds):
        # without supervision this blocks forever: the killed worker never reports DONE,
        # and the message loop waits for a count that can no longer be reached
        consumer = DatasetConsumer(num_proc=4, disable_tqdm=True, prefetch_factor=4)

        with pytest.raises(WorkerDiedError) as excinfo:
            consumer.consume(ds, finalizer=lambda b: _suicide_on_one_shard(b, 0))

        assert excinfo.value.deaths, "the error must name the worker that died"

    def test_the_error_says_it_was_killed_and_what_that_usually_means(self, ds):
        consumer = DatasetConsumer(num_proc=4, disable_tqdm=True, prefetch_factor=4)

        with pytest.raises(WorkerDiedError) as excinfo:
            consumer.consume(ds, finalizer=lambda b: _suicide_on_one_shard(b, 0))

        message = str(excinfo.value)
        assert "SIGKILL" in message
        assert "out-of-memory" in message
        # and it does not pretend to have a traceback it never got
        assert "no traceback" in message

    def test_the_exit_code_is_reported(self, ds):
        consumer = DatasetConsumer(num_proc=4, disable_tqdm=True, prefetch_factor=4)

        with pytest.raises(WorkerDiedError) as excinfo:
            consumer.consume(ds, finalizer=lambda b: _suicide_on_one_shard(b, 0))

        assert all(code == -signal.SIGKILL for _, code in excinfo.value.deaths)

    def test_a_healthy_run_is_untouched(self, ds, tmp_path):
        # The liveness check must not disturb a run where nothing dies. Counted from what
        # landed on disk rather than from a list: the workload runs in the workers, so a
        # list appended to there never reaches this process.
        out = str(tmp_path / "out")

        ArrowDatasetWriter(out, num_proc=4, disable_tqdm=True).write(ds)

        assert sorted(datasets.load_from_disk(out)["value"]) == sorted(
            s * 100 + i for s in range(4) for i in range(100)
        )
