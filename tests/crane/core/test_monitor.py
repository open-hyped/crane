import math

import pytest

from crane.core.monitor import ProgressMonitor
from crane.core.runners.base import WorkerProcessingStage, WorkerRole
from crane.core.utils import clock


class TestProgressMonitor:
    @pytest.fixture
    def monitor(self):
        return ProgressMonitor(5, 6, None, 10)

    def test_report_progress(self, monitor):
        # Simulate a worker's progress report over time
        rank = 0
        stage1 = WorkerProcessingStage.STREAM
        stage2 = WorkerProcessingStage.TRANSFORM
        stage3 = WorkerProcessingStage.FINALIZE

        monitor._mark_worker_ready(rank)
        monitor._mark_worker_busy(rank, WorkerRole.STANDALONE)

        now = clock()
        # Create mock progress reports
        report1 = {
            "timestamp": now + 1.0,
            "elapsed_time": 1.0,
            "num_samples": 100,
            "average_elapsed_time": {stage1: 0.5, stage2: 0.3, stage3: 0.4},
            "total_elapsed_time": {stage1: 0.5, stage2: 0.3, stage3: 0.4},
        }

        report2 = {
            "timestamp": now + 2.0,
            "elapsed_time": 1.0,
            "num_samples": 120,
            "average_elapsed_time": {stage1: 0.4, stage2: 0.35, stage3: 0.1},
            "total_elapsed_time": {stage1: 0.4, stage2: 0.35, stage3: 0.1},
        }

        # Call the _report_progress method with sample reports
        monitor._report_progress(rank, report1)
        monitor._report_progress(rank, report2)

        # Validate throughput calculation (samples/second)
        assert isinstance(monitor.samples_per_second, float)
        assert math.isclose(monitor.samples_per_second, 105.0, rel_tol=1e-2)

        # Validate average elapsed times
        averages = monitor.elapsed_time_averages([rank])
        assert math.isclose(averages[stage1], 0.45, rel_tol=1e-2)
        assert math.isclose(averages[stage2], 0.325, rel_tol=1e-2)
        assert math.isclose(averages[stage3], 0.25, rel_tol=1e-2)

    def test_is_stopping(self, monitor):
        # Initially, the monitor should not be stopping
        assert not monitor.is_stopping
        # Trigger the stopping event
        monitor._mark_as_stopping()
        assert monitor.is_stopping

    def test_is_done(self, monitor):
        # Initially, the monitor should not be stopping
        assert not monitor.is_done
        # Trigger the stopping event
        monitor._mark_as_done()
        assert monitor.is_done

    def test_pending_shards(self, monitor):
        assert monitor.pending_shards == {0, 1, 2, 3, 4}
        # Mark some shards as completed
        monitor._mark_shard_in_progress(0, 0)
        monitor._mark_shard_in_progress(1, 2)
        monitor._mark_shard_completed(2)
        assert monitor.pending_shards == {1, 3, 4}

    def test_any_pending_shards(self, monitor):
        # Initially, all shards should be pending
        assert monitor.any_pending_shards
        # Mark all shards as completed
        for shard_id in range(monitor.num_shards):
            monitor._mark_shard_completed(shard_id)
        # Check property
        assert not monitor.any_pending_shards

    def test_completed_shards(self, monitor):
        # Initially, no shards are completed
        assert monitor.completed_shards == set()

        # Mark some shards as completed
        for shard_id in (1, 4):
            monitor._mark_shard_in_progress(0, shard_id)
            monitor._mark_worker_completed(0)

        assert monitor.completed_shards == {1, 4}

    def test_any_worker_alive(self, monitor):
        # Initially, no workers are alive
        assert not monitor.any_worker_alive

        # Set one worker as alive
        monitor._mark_worker_ready(0)
        assert monitor.any_worker_alive

    def test_alive_workers(self, monitor):
        # Initially, no workers are alive
        assert monitor.alive_workers == set()
        # Set some workers as alive
        monitor._mark_worker_ready(1)
        monitor._mark_worker_ready(2)
        assert monitor.alive_workers == {1, 2}

    def test_idle_and_busy_workers(self, monitor):
        # mark workers ready
        monitor._mark_worker_ready(1)
        monitor._mark_worker_ready(2)
        # Assign roles to some workers
        monitor._mark_worker_busy(1, WorkerRole.CONSUMER)
        monitor._mark_worker_busy(2, WorkerRole.PRODUCER)
        assert monitor.idle_workers == {0, 3, 4, 5}
        assert monitor.busy_workers == {1, 2}
        # test worker roles
        assert monitor.get_workers_with_role(WorkerRole.CONSUMER) == {1}
        assert monitor.get_workers_with_role(WorkerRole.PRODUCER) == {2}
