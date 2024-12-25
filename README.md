# Hyped Crane

[![Tests](https://github.com/open-hyped/crane/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/open-hyped/crane/actions/workflows/tests.yml)
[![Linting](https://github.com/open-hyped/crane/actions/workflows/linting.yml/badge.svg?branch=main)](https://github.com/open-hyped/crane/actions/workflows/linting.yml)
[![Coverage Status](https://coveralls.io/repos/github/open-hyped/crane/badge.svg?branch=main)](https://coveralls.io/github/open-hyped/crane?branch=main)
[![PyPi version](https://badgen.net/pypi/v/hyped-crane/)](https://pypi.org/project/hyped-crane)
[![PyPi license](https://badgen.net/pypi/license/hyped-crane/)](https://pypi.org/project/hyped-crane/)

Lift (process) and place (write) data streams, seamlessly and in parallel.

Hyped Crane is a Python library designed to simplify working with HuggingFace `datasets`' iterable datasets. It provides powerful tools for applying transformations to data streams, handling parallel processing, and writing data in varios formats.


## Features

- **Streaming-Friendly Transformations**: Apply lazy, streaming-friendly transformations to iterable datasets without preloading data into memory.
- **Seamless Multiprocessing**: Effortlessly process and write datasets using multiple processes, improving performance on large datasets.
- **Easily Extendable**: Provides a straightforward interface to implement support for custom data formats.
- **Interoperability with Hugging Face Datasets**: Write datasets in formats directly loadable using HuggingFace’s `load_from_disk` function.

## Installation

To install the library from **PyPI**, run:

```bash
pip install hyped-crane
```

To install the library **from source**, clone run:

```bash
git clone https://github.com/open-hyped/crane.git
cd crane
pip install .
```

## Getting Started

Here’s a quick example to illustrate how Hyped Crane works

### Step 1: "Load" the Dataset

`crane` is designed to work seamlessly with HuggingFace’s iterable datasets. Let’s start by creating one:

```python
import datasets

# Create a dummy iterable dataset
dummy_data = [
    {"a": 0, "b": [1, 2, 3, 4]},
    {"a": 1, "b": [5, 6]},
    {"a": 1, "b": [7, 8, 9, 10]}
]
ds = datasets.Dataset.from_list(dummy_data)
ds = ds.to_iterable_dataset()
```

### Step 2: Apply a Lazy Transformation

Transformations on iterable datasets are applied lazily. That means the data isn’t processed until it’s actually read:

```python
# Apply a transformation to compute the maximum of list "b"
ds = ds.map(lambda x: {"max(b)": max(x["b"])})
```

### Step 3: Write the Dataset to Disk

Use `crane`’s `ArrowDatasetWriter` to save the transformed dataset to disk. You can enable multiprocessing to speed up the transformation and writing processes:

```python
from crane import ArrowDatasetWriter

# Write the transformed dataset to disk with multiprocessing
writer = ArrowDatasetWriter("data", overwrite=True, num_proc=3)
writer.write(ds)
```

**Key Benefits**:
- **Data-Parallel Transformations**: The transformations defined by `map` operations are moved into the workers, allowing transformation workload to be evenly distributed.
- **Efficient Writing**: Each worker writes its own shard to disk in parallel, reducing bottlenecks in **I/O operations**.

`crane` handles worker communication, task distribution, and writing operations, so you can focus on defining your transformation logic without worrying about parallelization details.

**Note**: Datasets saved with the `ArrowDatasetWriter` are fully compatible with HuggingFace’s `load_from_disk` function. You can reload the dataset and continue working with it:

```python
# Reload the dataset from disk
ds = datasets.load_from_disk("data")
```


## Contributions

Contributions are welcome! Feel free to submit a pull request or open an issue to discuss your ideas.