"""The number of samples of each shard, recorded in the dataset info as the split's
`shard_lengths`, and their sum as its `num_examples`."""
import json
import os

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import IterableDataset
from datasets.iterable_dataset import ArrowExamplesIterable

from crane import ArrowDatasetWriter, JsonDatasetWriter, ParquetDatasetWriter
from crane.core.sharding import ShardingStrategy

FEATURES = datasets.Features({"id": datasets.Value("int64")})


def _tables(sizes: tuple[int, ...]):
    """Tables of `sizes` rows; a tuple, so that `datasets` keeps them in one shard."""
    start = 0
    for i, size in enumerate(sizes):
        yield i, pa.table({"id": list(range(start, start + size))})
        start += size


def _num_rows(path: str) -> int:
    """The rows of the shard file at `path`, read back."""
    if path.endswith(".json"):
        with open(path) as f:
            return sum(1 for _ in f)
    if path.endswith(".parquet"):
        return pq.read_metadata(path).num_rows
    return sum(batch.num_rows for batch in pa.ipc.open_stream(pa.memory_map(path)))


@pytest.mark.parametrize(
    "writer_type", [JsonDatasetWriter, ParquetDatasetWriter, ArrowDatasetWriter]
)
@pytest.mark.parametrize("num_proc", [1, 3])
def test_the_info_records_the_samples_of_each_shard(tmp_path, writer_type, num_proc):
    ds = IterableDataset(
        ArrowExamplesIterable(_tables, {"sizes": (7, 30, 1, 12, 50) * 10}),
        info=datasets.DatasetInfo(features=FEATURES),
    )
    save_dir = os.path.join(tmp_path, "data")

    writer_type(
        save_dir,
        num_proc=num_proc,
        sharding_strategy=ShardingStrategy.SAMPLE_COUNT,
        max_shard_size=100,
        write_batch_size=16,
        prefetch_factor=16,
        disable_tqdm=True,
    ).write(ds)

    shards = sorted(f for f in os.listdir(save_dir) if f.startswith("shard-"))
    split = datasets.DatasetInfo.from_directory(save_dir).splits["train"]
    assert len(shards) > 1
    assert split.shard_lengths == [_num_rows(os.path.join(save_dir, f)) for f in shards]
    assert split.num_examples == 1000


def test_the_info_replaces_the_splits_of_the_source(tmp_path):
    # the source's own splits describe what was read, not what was written
    info = datasets.DatasetInfo(
        features=FEATURES,
        splits=datasets.SplitDict({"train": datasets.SplitInfo(name="train", num_examples=12345)}),
    )
    ds = IterableDataset(
        ArrowExamplesIterable(_tables, {"sizes": (5, 5)}), info=info, split=datasets.Split.TRAIN
    )
    save_dir = os.path.join(tmp_path, "data")

    JsonDatasetWriter(save_dir, num_proc=1, disable_tqdm=True).write(ds)

    with open(os.path.join(save_dir, datasets.config.DATASET_INFO_FILENAME)) as f:
        splits = json.load(f)["splits"]
    assert splits == {
        "train": {
            "name": "train",
            "num_bytes": 0,
            "num_examples": 10,
            "shard_lengths": [10],
            "dataset_name": None,
        }
    }
