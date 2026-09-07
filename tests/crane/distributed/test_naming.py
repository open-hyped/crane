import dill

from crane.core.naming import ShardName
from crane.distributed.core.naming import JobShardName


class TestJobShardName:
    def test_the_name_carries_the_job_index(self):
        assert JobShardName(7)(42, "arrow") == "shard-007-00042.arrow"

    def test_jobs_cannot_collide_whatever_they_write(self):
        # every job numbers its own shards from zero; only the job index keeps them apart
        names = {JobShardName(j)(i, "arrow") for j in range(4) for i in range(10)}

        assert len(names) == 40

    def test_it_is_a_shard_name(self):
        # so that a writer takes it through `_set_shard_name` like any other policy
        assert isinstance(JobShardName(0), ShardName)

    def test_it_survives_being_sent_to_a_worker(self):
        assert dill.loads(dill.dumps(JobShardName(7)))(42, "arrow") == "shard-007-00042.arrow"
