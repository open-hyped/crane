import json
import os
from abc import ABC, abstractmethod
from typing import Any, Iterable

import pytest
from datasets import Dataset

from crane.core.utils import chdir
from crane.core.writer import BaseDatasetWriter


class BaseTestDatasetWriter(ABC):
    dataset: Dataset
    writer_type: type[BaseDatasetWriter]
    writer_args: dict[str, Any] = {}
    # Writers are exercised with more than one process by default: multiprocessing is what
    # the writers are configured for out of the box, and pinning every writer test to a
    # single process left the multi-process path completely untested. Subclasses whose
    # expectations depend on a specific shard layout narrow this down.
    supported_num_procs: tuple[int, ...] = (1, 2)

    @pytest.fixture(params=[1, 2])
    def num_proc(self, request) -> int:
        if request.param not in type(self).supported_num_procs:
            pytest.skip(f"{type(self).__name__} does not support num_proc={request.param}")
        # kept on the instance so the checks in execute_test can take the number of
        # processes into account; pytest builds a fresh instance per test
        self.num_proc = request.param
        return request.param

    @pytest.fixture
    def writer(self, tmp_path, num_proc) -> BaseDatasetWriter:
        cls = type(self)
        # build keyword arguments
        kwargs = cls.writer_args.copy()
        kwargs["save_dir"] = os.path.join(tmp_path, "data")
        kwargs["num_proc"] = num_proc
        # create writer instance
        return cls.writer_type(**kwargs)

    def assert_same_samples(self, actual: Iterable[Any], expected: Iterable[Any]) -> None:
        """Assert that two sample streams hold the same samples.

        Each process writes its own shards, and which process picks up which part of the
        dataset is decided at runtime, so the order the samples come back in is only
        meaningful for a single-process write.

        Args:
            actual (Iterable[Any]): The samples read back from the written dataset.
            expected (Iterable[Any]): The samples the dataset was written from.
        """
        actual, expected = list(actual), list(expected)

        if self.num_proc > 1:

            def key(sample: Any) -> str:
                return json.dumps(sample, sort_keys=True, default=str)

            actual, expected = sorted(actual, key=key), sorted(expected, key=key)

        assert actual == expected

    @abstractmethod
    def execute_test(self) -> None:
        ...

    def test_case(self, writer):
        # write dataset to disk
        writer.write(type(self).dataset)
        # check save directory
        with chdir(writer.save_dir):
            self.execute_test()
