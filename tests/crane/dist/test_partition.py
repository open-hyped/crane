import datasets
import pytest

from crane.dist.core.partition import num_jobs_for, select_shards


def _dataset(num_shards: int, rows_per_shard: int = 4) -> datasets.IterableDataset:
    """A dataset whose samples say which shard they came from."""

    def gen(shards):
        for shard in shards:
            for i in range(rows_per_shard):
                yield {"shard": shard, "value": shard * rows_per_shard + i}

    return datasets.IterableDataset.from_generator(
        gen, gen_kwargs={"shards": list(range(num_shards))}
    )


class TestNumJobsFor:
    def test_leaves_a_sensible_count_alone(self):
        assert num_jobs_for(_dataset(8), 4) == 4

    def test_clamps_to_the_shard_count(self):
        # more jobs than shards would leave jobs holding an allocation and no work
        assert num_jobs_for(_dataset(3), 16) == 3

    def test_rejects_a_non_positive_count(self):
        with pytest.raises(ValueError, match="at least one"):
            num_jobs_for(_dataset(4), 0)


class TestSelectShards:
    @pytest.mark.parametrize("num_jobs", [1, 2, 3, 4, 8])
    def test_every_sample_lands_in_exactly_one_job(self, num_jobs):
        ds = _dataset(8)
        expected = [s["value"] for s in ds]

        seen = []
        for job_index in range(num_jobs):
            seen += [s["value"] for s in select_shards(ds, num_jobs, job_index)]

        # no sample is dropped and none is processed twice
        assert sorted(seen) == sorted(expected)
        assert len(seen) == len(expected)

    def test_split_is_strided(self):
        ds = _dataset(6)
        shards = {s["shard"] for s in select_shards(ds, 3, 1)}
        assert shards == {1, 4}

    def test_shard_count_reflects_the_share(self):
        # the job's dataset must look like an ordinary one to the runners
        assert select_shards(_dataset(8), 4, 0).n_shards == 2
        assert select_shards(_dataset(9), 4, 0).n_shards == 3

    def test_transforms_survive_the_split(self):
        ds = _dataset(4).map(lambda x: {"doubled": x["value"] * 2})
        samples = list(select_shards(ds, 2, 0))
        assert all(s["doubled"] == s["value"] * 2 for s in samples)
        assert len(samples) > 0

    def test_features_survive_the_split(self):
        ds = _dataset(4)
        assert select_shards(ds, 2, 0).info is not None

    def test_rejects_an_index_outside_the_run(self):
        with pytest.raises(ValueError, match="out of range"):
            select_shards(_dataset(4), 2, 2)

    def test_rejects_a_job_with_no_shards(self):
        # guarded rather than silently producing an empty job
        with pytest.raises(ValueError, match="no shards"):
            select_shards(_dataset(2), 4, 3)
