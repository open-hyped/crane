"""How a distributed run names its shards.

The jobs of a run write into one save directory, and shard ids are counted within a job, so
a job's shards would otherwise land on its siblings' files. The job index in the name is
what makes that impossible, whatever a writer does with the path it is handed.
"""

from dataclasses import dataclass

from ...core.naming import ShardName


@dataclass(frozen=True)
class JobShardName(ShardName):
    """Names the shards of one job of a distributed run.

    Installed on the writer by the job's entrypoint rather than read from the environment,
    so that a stale export cannot quietly change the names a local write produces.
    """

    job_index: int
    """The index of the job doing the writing."""

    def __call__(self, shard_id: int, ext: str) -> str:
        """Build the file name of a shard, unique across the jobs of the run.

        Args:
            shard_id (int): The id of the shard, counted within this job.
            ext (str): The file extension the writer produces, without a leading dot.

        Returns:
            str: The shard's file name, relative to the save directory.
        """
        return f"shard-{self.job_index:03}-{shard_id:05}.{ext}"
