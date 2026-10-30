"""A dataset of several shards, cut to a window with `skip` and `take`, is written as that window.

`datasets` splits the count of `skip` and `take` among the shards when a dataset is sharded, so
that the workers of a PyTorch DataLoader skip and take `n` between them. A runner that reads the
shards one by one must still write the first `n` samples of the dataset, not `n / num_shards` of
each shard.
"""

import json
import logging
import os

import pytest
from datasets import IterableDataset

from crane import JsonDatasetWriter
from crane.core.iterables import shards_to_read
from crane.core.runners.multi_process_runner import _is_separable

SHARDS = [(0, 256), (256, 300)]
"""Two shards of uneven size, their samples numbered on from one to the next."""


def _samples(shards):
    for start, stop in shards:
        for i in range(start, stop):
            yield {"obj": i}


def _dataset() -> IterableDataset:
    """The samples 0 to 299, in two shards: a list in `gen_kwargs` is a shard per item."""
    ds = IterableDataset.from_generator(_samples, gen_kwargs={"shards": SHARDS})
    assert ds.n_shards == len(SHARDS)
    return ds


def _written(save_dir: str) -> list[int]:
    """The samples written to `save_dir`, all its shards."""
    samples = []
    for name in sorted(os.listdir(save_dir)):
        if name.startswith("shard-") and name.endswith(".json"):
            with open(os.path.join(save_dir, name)) as f:
                samples += [json.loads(line)["obj"] for line in f if line.strip()]
    return samples


@pytest.mark.parametrize("num_proc", [1, 2])
@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (lambda ds: ds, range(0, 300)),
        (lambda ds: ds.take(250), range(0, 250)),
        (lambda ds: ds.skip(100), range(100, 300)),
        (lambda ds: ds.skip(100).take(150), range(100, 250)),
        # the window begins in the second shard
        (lambda ds: ds.skip(260).take(20), range(260, 280)),
    ],
    ids=["all", "take", "skip", "skip-take", "skip-into-second-shard"],
)
def test_window_of_sharded_dataset(tmp_path, num_proc, window, expected):
    save_dir = os.path.join(tmp_path, "data")
    JsonDatasetWriter(save_dir=save_dir, num_proc=num_proc).write(window(_dataset()))
    assert sorted(_written(save_dir)) == list(expected)


def test_window_is_written_in_order_by_one_process(tmp_path):
    save_dir = os.path.join(tmp_path, "data")
    JsonDatasetWriter(save_dir=save_dir, num_proc=1).write(_dataset().skip(100).take(200))
    assert _written(save_dir) == list(range(100, 300))


@pytest.mark.parametrize(
    "window", [lambda ds: ds.skip(1), lambda ds: ds.take(1)], ids=["skip", "take"]
)
def test_window_is_read_as_one_shard_with_a_warning(caplog, window):
    with caplog.at_level(logging.WARNING, logger="crane"):
        assert shards_to_read(window(_dataset())._ex_iterable) == 1
    assert any("read as one shard" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("build", "shards"),
    [
        # not cut: every shard is read on its own
        (lambda: _dataset(), len(SHARDS)),
        # cut, but of one shard only: nothing to count across
        (
            lambda: IterableDataset.from_generator(_samples, gen_kwargs={"shards": [(0, 10)]}).take(
                5
            ),
            1,
        ),
    ],
    ids=["not-cut", "one-shard"],
)
def test_no_warning_where_the_shards_need_not_be_read_as_one(caplog, build, shards):
    with caplog.at_level(logging.WARNING, logger="crane"):
        assert shards_to_read(build()._ex_iterable) == shards
    assert not caplog.records


def test_window_stays_with_the_data_source():
    """The multi-process runner separates the processing steps off the data source into its
    workers; skip and take must stay with the source, read once, not be counted per worker."""
    ds = _dataset().map(lambda sample: sample).skip(10).take(5)
    assert not _is_separable(ds._ex_iterable)
    assert not _is_separable(ds._ex_iterable.ex_iterable)
