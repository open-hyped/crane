import os
from abc import ABC, abstractmethod
from typing import Any

import pytest
from datasets import Dataset

from crane.core.utils import chdir
from crane.core.writer import BaseDatasetWriter


class BaseTestDatasetWriter(ABC):
    dataset: Dataset
    writer_type: type[BaseDatasetWriter]
    writer_args: dict[str, Any] = {}

    @pytest.fixture
    def writer(self, tmp_path) -> BaseDatasetWriter:
        cls = type(self)
        # build keyword arguments
        kwargs = cls.writer_args.copy()
        kwargs["save_dir"] = os.path.join(tmp_path, "data")
        kwargs["num_proc"] = 1
        # create writer instance
        return cls.writer_type(**kwargs)

    @abstractmethod
    def execute_test(self) -> None:
        ...

    def test_case(self, writer):
        # write dataset to disk
        writer.write(type(self).dataset)
        # check save directory
        with chdir(writer.save_dir):
            self.execute_test()
