import pyarrow as pa
import pytest

from crane.core.batching import BatchBuffer


def table(rows: int) -> pa.Table:
    return pa.table({"x": list(range(rows))})


def columns(rows: int) -> dict:
    """The other format a writer receives: a dictionary of columns."""
    return {"x": list(range(rows))}


class TestBatchBuffer:
    def test_holds_batches_until_there_are_enough_rows(self):
        buffer = BatchBuffer(size=10)

        assert buffer.add(table(4)) == []
        assert buffer.add(table(4)) == []

    @pytest.mark.parametrize("arriving", [1, 3, 8, 250])
    def test_what_comes_back_is_the_asked_for_size_whatever_arrives(self, arriving):
        # the whole point: flushing the pile instead would give 504 for batches of 8 and
        # 512 for batches of 256, so the layout would still depend on the arrival size
        buffer = BatchBuffer(size=500)
        sizes = []

        for _ in range(1000 // arriving):
            sizes += [t.num_rows for t in buffer.add(table(arriving))]

        assert set(sizes) <= {500}

    def test_each_batch_handed_back_is_a_single_record_batch(self):
        # concatenating alone would leave one chunk per batch held, and a writer emits a
        # record batch per chunk
        buffer = BatchBuffer(size=3)
        buffer.add(table(1))
        buffer.add(table(1))

        (out,) = buffer.add(table(1))

        assert len(out.to_batches()) == 1

    def test_one_arrival_can_complete_several_batches(self):
        buffer = BatchBuffer(size=10)

        out = buffer.add(table(35))

        assert [t.num_rows for t in out] == [10, 10, 10]

    def test_taking_empties_the_buffer(self):
        buffer = BatchBuffer(size=10)
        buffer.add(table(4))

        assert buffer.take().num_rows == 4
        assert buffer.take() is None

    def test_taking_an_empty_buffer_is_not_an_error(self):
        assert BatchBuffer(size=10).take() is None

    def test_rows_are_never_lost_across_a_run_of_batches(self):
        buffer = BatchBuffer(size=7)
        written = sum(t.num_rows for _ in range(20) for t in buffer.add(table(3)))

        remainder = buffer.take()

        assert written + (remainder.num_rows if remainder is not None else 0) == 60

    def test_a_size_of_one_writes_every_row_through(self):
        buffer = BatchBuffer(size=1)

        assert [t.num_rows for t in buffer.add(table(4))] == [1, 1, 1, 1]


class TestColumnBatches:
    """The other format a writer receives, which the JSON writer uses."""

    def test_columns_are_regrouped_to_the_asked_for_size(self):
        buffer = BatchBuffer(size=10)
        sizes = []

        for _ in range(30):
            sizes += [len(b["x"]) for b in buffer.add(columns(1))]

        assert sizes == [10, 10, 10]

    def test_a_remainder_comes_back_whole(self):
        buffer = BatchBuffer(size=10)
        buffer.add(columns(4))

        assert len(buffer.take()["x"]) == 4

    def test_rows_keep_their_order_and_values(self):
        buffer = BatchBuffer(size=3)
        out = []

        for start in (0, 3, 6):
            out += buffer.add({"x": [start, start + 1, start + 2]})

        assert [b["x"] for b in out] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_an_empty_batch_changes_nothing(self):
        # a map that filters everything out of a batch still reaches the writer
        buffer = BatchBuffer(size=3)

        assert buffer.add({"x": []}) == []
        assert buffer.take() is None
