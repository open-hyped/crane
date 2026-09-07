"""How the files a writer produces are named.

Naming belongs to the write path rather than to an individual writer: a writer says what
extension its files carry and what to put inside them, never what they are called. Keeping
the choice in one place is what lets a launcher which runs several writers against a single
directory guarantee that their files cannot collide, whatever those writers do with the
paths they are handed - a guarantee a convention about shard ids could not give, since the
name would still be chosen downstream.

A launcher that needs different names supplies its own policy through
:func:`BaseDatasetWriter._set_shard_name`. The policies are dataclasses rather than
closures so that they survive the trip into a worker process along with the writer holding
them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShardName(object):
    """Names the shard files of a run.

    Shard ids are counted within one write, so these names are unique within a run and no
    further.
    """

    def __call__(self, shard_id: int, ext: str) -> str:
        """Build the file name of a shard.

        Args:
            shard_id (int): The id of the shard, counted within this run.
            ext (str): The file extension the writer produces, without a leading dot.

        Returns:
            str: The shard's file name, relative to the save directory.
        """
        return f"shard-{shard_id:05}.{ext}"
