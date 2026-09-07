"""Splitting a dataset's shards across distributed jobs.

This module provides :func:`select_shards`, which restricts a dataset to the shards
belonging to a single job of a distributed run.
"""

import logging

from datasets import IterableDataset

logger = logging.getLogger(__name__)


def num_jobs_for(ds: IterableDataset, num_jobs: int) -> int:
    """Clamp a requested job count to the number of shards the dataset actually has.

    A job without a shard has nothing to do but still occupies an allocation, so asking
    for more jobs than there are shards is reduced to the useful number rather than
    honored.

    Args:
        ds (IterableDataset): The dataset to be distributed.
        num_jobs (int): The requested number of jobs.

    Returns:
        int: The number of jobs to actually submit.

    Raises:
        ValueError: If :code:`num_jobs` is not positive.
    """
    if num_jobs < 1:
        raise ValueError(f"`num_jobs` must be at least one, got {num_jobs}.")

    if num_jobs > ds.n_shards:
        logger.warning(
            f"Requested {num_jobs} jobs but the dataset has only {ds.n_shards} shard(s); "
            f"submitting {ds.n_shards}. Shards are the unit of distribution, so the extra "
            f"jobs would occupy an allocation without any work to do."
        )
        num_jobs = ds.n_shards

    if num_jobs == 1:
        logger.warning(
            "The run is distributed across a single job, so it will not run any faster "
            "than a local one. A dataset is distributed by its shards, and this one "
            f"exposes {ds.n_shards}."
        )

    return num_jobs


def select_shards(ds: IterableDataset, num_jobs: int, job_index: int) -> IterableDataset:
    """Restrict a dataset to the shards belonging to one job.

    Job :code:`j` of :code:`num_jobs` takes shards :code:`j, j + num_jobs, ...` - strided
    rather than contiguous, so a dataset whose shards grow or shrink along their order does
    not hand one job all the large ones.

    The result is an ordinary dataset whose :code:`n_shards` is the job's own share, which
    is what lets the runners split it further across processes without knowing that
    anything larger is going on.

    **Note**: This is deliberately not :func:`datasets.distributed.split_dataset_by_node`.
    That falls back to keeping one example out of :code:`world_size` and skipping the rest
    whenever the shard count is not a multiple of the job count, in which case every job
    reads the whole dataset. It also works by setting :code:`IterableDataset._distributed`,
    which :func:`DynamicMultiprocessingRunner._prepare_dataset` discards when it rebuilds
    the dataset around the separated iterable.

    Args:
        ds (IterableDataset): The dataset to restrict.
        num_jobs (int): The total number of jobs the run is split across.
        job_index (int): The index of the job to select shards for.

    Returns:
        IterableDataset: A dataset over this job's shards only.

    Raises:
        ValueError: If :code:`job_index` does not address a shard of the dataset.
    """
    if not (0 <= job_index < num_jobs):
        raise ValueError(f"Job index {job_index} is out of range for {num_jobs} job(s).")

    if job_index >= ds.n_shards:
        raise ValueError(
            f"Job {job_index} has no shards to process: the dataset has {ds.n_shards} "
            f"shard(s) and the run was split {num_jobs} ways."
        )

    ex_iterable = ds._ex_iterable.shard_data_sources(
        num_shards=num_jobs, index=job_index, contiguous=False
    )
    # Rebuilt rather than mutated: `info` carries the features every writer needs to build
    # its arrow schema, and `formatting` decides whether the workload sees arrow or python.
    shard_ds = IterableDataset(
        ex_iterable=ex_iterable,
        info=ds.info,
        split=ds.split,
        formatting=ds._formatting,
    )
    logger.info(f"Job {job_index} of {num_jobs} took {shard_ds.n_shards} of {ds.n_shards} shards.")
    return shard_ds
