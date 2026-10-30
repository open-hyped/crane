"""The indices a function given them (`with_indices=True`) gets when a dataset is written.

Each worker counts the samples it processes, from 0 and on across every shard and role it
takes, as a worker of a PyTorch `DataLoader` does. With one worker these are the positions in the
dataset, as in one pass over it with `datasets`.
"""
import collections
import json
import logging
import os
import time

import datasets
import pyarrow as pa
import pytest
from datasets import IterableDataset
from datasets.iterable_dataset import ArrowExamplesIterable

from crane import JsonDatasetWriter

BATCH_SIZE = 16


def _tables(shards: list[tuple[int, int]]):
    """Each shard's tables of `BATCH_SIZE` rows, `id` the row's position across all shards; a
    list, so that `datasets` makes a shard of each of its items."""
    for start, num_tables in shards:
        for table in range(num_tables):
            first = start + table * BATCH_SIZE
            yield f"{start}_{table}", pa.table({"id": list(range(first, first + BATCH_SIZE))})


def _source(num_shards: int, tables_per_shard: int = 20) -> IterableDataset:
    rows_per_shard = tables_per_shard * BATCH_SIZE
    shards = [(shard * rows_per_shard, tables_per_shard) for shard in range(num_shards)]
    return IterableDataset(ArrowExamplesIterable(_tables, {"shards": shards}))


def _keep(batch, indices):
    """Drop every third sample, by its position."""
    return [index % 3 != 0 for index in indices]


def _record(batch, indices):
    """The index of each sample, and the worker it was processed by; slow enough that more
    than one worker gets a share of the samples."""
    time.sleep(0.002)
    return {"idx": indices, "pid": [os.getpid()] * len(indices)}


FEATURES = datasets.Features(
    {"id": datasets.Value("int64"), "idx": datasets.Value("int64"), "pid": datasets.Value("int64")}
)


def _write(ds: IterableDataset, tmp_path, num_proc: int) -> list[dict]:
    save_dir = os.path.join(tmp_path, "data")
    JsonDatasetWriter(
        save_dir,
        num_proc=num_proc,
        prefetch_factor=BATCH_SIZE,
        write_batch_size=BATCH_SIZE,
        disable_tqdm=True,
    ).write(ds)
    shards = sorted(f for f in os.listdir(save_dir) if f.startswith("shard-"))
    rows = []
    for shard in shards:
        with open(os.path.join(save_dir, shard)) as f:
            rows += map(json.loads, f)
    return rows


@pytest.mark.parametrize("num_shards", [1, 8])
def test_one_worker_counts_on_across_the_shards(tmp_path, num_shards):
    ds = _source(num_shards).map(_record, with_indices=True, batched=True, features=FEATURES)

    rows = _write(ds, tmp_path, num_proc=1)

    assert [row["idx"] for row in rows] == [row["id"] for row in rows]
    assert len(rows) == num_shards * 20 * BATCH_SIZE


def test_one_worker_counts_after_a_filter_as_datasets_does(tmp_path):
    # the map's indices are the positions among the samples the filter kept, as in one pass
    ds = (
        _source(8)
        .filter(_keep, with_indices=True, batched=True)
        .map(_record, with_indices=True, batched=True, features=FEATURES)
    )
    expected = [(row["id"], row["idx"]) for row in ds]

    rows = _write(ds, tmp_path, num_proc=1)

    assert [(row["id"], row["idx"]) for row in rows] == expected


@pytest.mark.parametrize("num_shards", [1, 8])
def test_each_worker_counts_its_own_samples(tmp_path, num_shards):
    ds = _source(num_shards).map(_record, with_indices=True, batched=True, features=FEATURES)

    rows = _write(ds, tmp_path, num_proc=3)

    assert len(rows) == num_shards * 20 * BATCH_SIZE
    by_worker = collections.defaultdict(list)
    for row in rows:
        by_worker[row["pid"]].append(row["idx"])
    # each worker's shards hold its samples in the order it processed them
    for indices in by_worker.values():
        assert indices == list(range(len(indices)))


def _identity(batch):
    return batch


def _keep_even(batch):
    return [value % 2 == 0 for value in batch["id"]]


@pytest.mark.parametrize(
    "build, num_proc, expected",
    [
        # indices of one worker, before any filter: the positions in the dataset
        (lambda ds: ds.map(_record, with_indices=True, batched=True), 1, []),
        (lambda ds: ds.filter(_keep, with_indices=True, batched=True), 1, []),
        (lambda ds: ds.map(_identity, batched=True), 3, []),
        # several workers each count their own
        (lambda ds: ds.map(_record, with_indices=True, batched=True), 3, ["processes"]),
        # the indices count what the filter kept
        (
            lambda ds: ds.filter(_keep_even, batched=True).map(
                _record, with_indices=True, batched=True
            ),
            1,
            ["after a filter"],
        ),
        (
            lambda ds: ds.filter(_keep_even, batched=True).filter(
                _keep, with_indices=True, batched=True
            ),
            3,
            ["processes", "after a filter"],
        ),
    ],
)
def test_a_warning_where_the_indices_are_not_the_positions(caplog, build, num_proc, expected):
    from crane.core.iterables import ExampleIndices

    with caplog.at_level(logging.WARNING, logger="crane"):
        ExampleIndices.warn(build(_source(2))._ex_iterable, num_proc)

    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == len(expected)
    for warning, phrase in zip(warnings, expected):
        assert phrase in warning


def test_the_writer_warns(tmp_path, caplog):
    ds = _source(1).map(_record, with_indices=True, batched=True, features=FEATURES)

    with caplog.at_level(logging.WARNING, logger="crane"):
        _write(ds, tmp_path, num_proc=2)

    assert any("with_indices=True" in record.getMessage() for record in caplog.records)
