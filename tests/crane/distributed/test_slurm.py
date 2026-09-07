import subprocess
from unittest.mock import MagicMock, patch

import pytest

from crane.distributed.core.base import JobState, RunSpec
from crane.distributed.slurm import MAX_NUM_JOBS, Slurm, SlurmNotAvailableError


@pytest.fixture
def spec(tmp_path) -> RunSpec:
    return RunSpec(
        run_id="out-abc123",
        run_dir=str(tmp_path / "run"),
        num_jobs=4,
        num_shards=16,
        save_dir=str(tmp_path / "out"),
        python="/usr/bin/python",
        needs_finalize=True,
        backend={"name": "slurm", "num_jobs": 4},
    )


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class TestJobCount:
    def test_accepts_a_usable_count(self):
        assert Slurm(num_jobs=MAX_NUM_JOBS).num_jobs == MAX_NUM_JOBS

    def test_rejects_more_jobs_than_slurm_will_take(self):
        # rejected here rather than by the controller after a payload has been serialized
        with pytest.raises(ValueError, match=f"more than the {MAX_NUM_JOBS} jobs"):
            Slurm(num_jobs=MAX_NUM_JOBS + 1)

    def test_the_error_says_what_to_do_instead(self):
        with pytest.raises(ValueError, match="fewer jobs, each taking more"):
            Slurm(num_jobs=5000)

    @pytest.mark.parametrize("num_jobs", [0, -1])
    def test_rejects_a_non_positive_count(self, num_jobs):
        with pytest.raises(ValueError, match="at least one"):
            Slurm(num_jobs=num_jobs)


class TestAvailability:
    def test_submit_refuses_without_slurm(self, spec):
        # the error has to say what is missing and what to do instead
        with patch("crane.distributed.slurm.shutil.which", return_value=None):
            with pytest.raises(SlurmNotAvailableError, match="sbatch"):
                Slurm(num_jobs=4).submit(spec)

    def test_error_points_at_the_local_alternative(self, spec):
        with patch("crane.distributed.slurm.shutil.which", return_value=None):
            with pytest.raises(SlurmNotAvailableError, match="`write` rather than `submit`"):
                Slurm(num_jobs=4).poll(spec, ["1"])


class TestScript:
    def test_directives_carry_the_run_marker(self, spec):
        script = Slurm(num_jobs=4)._script(spec, "logs", ["run"], array=True)
        # the marker `attach` reads back, independent of the job name
        assert "#SBATCH --comment=crane:out-abc123" in script

    def test_array_covers_every_job(self, spec):
        script = Slurm(num_jobs=4)._script(spec, "logs", ["run"], array=True)
        assert "#SBATCH --array=0-3" in script

    def test_max_concurrent_throttles_the_array(self, spec):
        script = Slurm(num_jobs=4, max_concurrent=2)._script(spec, "logs", ["run"], array=True)
        assert "#SBATCH --array=0-3%2" in script

    def test_finalize_script_has_no_array(self, spec):
        script = Slurm(num_jobs=4)._script(spec, "logs", ["run"], array=False)
        assert "--array" not in script
        assert "finalize" in script

    def test_unset_options_are_omitted(self, spec):
        script = Slurm(num_jobs=4)._script(spec, "logs", ["run"], array=True)
        assert "--partition" not in script
        assert "--mem" not in script

    def test_named_options_are_rendered(self, spec):
        backend = Slurm(num_jobs=4, partition="batch", time="00:10:00", cpus_per_task=2, mem="4GB")
        script = backend._script(spec, "logs", ["run"], array=True)
        assert "#SBATCH --partition=batch" in script
        assert "#SBATCH --time=00:10:00" in script
        assert "#SBATCH --cpus-per-task=2" in script
        assert "#SBATCH --mem=4GB" in script

    def test_sbatch_args_are_passed_through(self, spec):
        backend = Slurm(num_jobs=4, sbatch_args={"--gres": "gpu:1"})
        assert "#SBATCH --gres=gpu:1" in backend._script(spec, "logs", ["run"], array=True)

    def test_setup_and_env_precede_the_command(self, spec):
        backend = Slurm(num_jobs=4, setup=["module load python"], env={"HF_HOME": "/cache"})
        script = backend._script(spec, "logs", ["the-command"], array=True)
        assert script.index("module load python") < script.index("the-command")
        assert script.index("export HF_HOME=/cache") < script.index("the-command")

    def test_job_announces_itself_before_anything_heavy(self, spec):
        # a job killed during startup must not leave an empty log
        script = Slurm(num_jobs=4)._script(spec, "logs", ["the-command"], array=True)
        assert "export PYTHONUNBUFFERED=1" in script
        assert script.index('echo "crane: run out-abc123') < script.index("the-command")

    def test_dry_run_submits_nothing(self, spec, capsys):
        with patch("crane.distributed.slurm.shutil.which", return_value="/usr/bin/sbatch"):
            with patch("crane.distributed.slurm._run") as run:
                assert Slurm(num_jobs=4, dry_run=True).submit(spec) == []
                run.assert_not_called()
        assert "#SBATCH" in capsys.readouterr().out


class TestSubmit:
    @pytest.fixture(autouse=True)
    def _slurm_available(self):
        with patch("crane.distributed.slurm.shutil.which", return_value="/usr/bin/sbatch"):
            yield

    def test_submits_array_then_dependent_finalize(self, spec):
        side_effect = [_completed("111"), _completed("222")]
        with patch("crane.distributed.slurm._run", side_effect=side_effect) as run:
            assert Slurm(num_jobs=4).submit(spec) == ["111", "222"]

        # the finalize job must not start unless every worker succeeded
        assert "--dependency=afterok:111" in run.call_args_list[1].args[0]

    def test_no_finalize_job_when_nothing_needs_finalizing(self, spec, tmp_path):
        spec = RunSpec(**(spec.__dict__ | {"needs_finalize": False}))
        with patch("crane.distributed.slurm._run", side_effect=[_completed("111")]) as run:
            assert Slurm(num_jobs=4).submit(spec) == ["111"]
        assert run.call_count == 1

    def test_parses_only_the_job_id(self, spec):
        # `sbatch --parsable` appends `;cluster` on a federated setup
        side_effect = [_completed("111;cluster"), _completed("222")]
        with patch("crane.distributed.slurm._run", side_effect=side_effect):
            assert Slurm(num_jobs=4).submit(spec)[0] == "111"


class TestPoll:
    @pytest.fixture(autouse=True)
    def _slurm_available(self):
        with patch("crane.distributed.slurm.shutil.which", return_value="/usr/bin/squeue"):
            yield

    def test_reads_array_task_states(self, spec):
        with patch(
            "crane.distributed.slurm._run",
            side_effect=[_completed("0;RUNNING\n2;PENDING\n"), _completed("")],
        ):
            view = Slurm(num_jobs=4).poll(spec, ["111", "222"])

        assert view.active_jobs == {0: JobState.RUNNING, 2: JobState.PENDING}
        assert view.finalize_pending is False

    def test_a_job_out_of_the_queue_is_not_active(self, spec):
        # squeue exits non-zero for a job it no longer knows, which is an answer
        with patch(
            "crane.distributed.slurm._run", side_effect=[_completed("", 1), _completed("", 1)]
        ):
            view = Slurm(num_jobs=4).poll(spec, ["111", "222"])
        assert view.active_jobs == {}

    def test_finalize_pending_is_reported(self, spec):
        with patch(
            "crane.distributed.slurm._run", side_effect=[_completed(""), _completed("PENDING\n")]
        ):
            assert Slurm(num_jobs=4).poll(spec, ["111", "222"]).finalize_pending

    def test_transient_states_count_as_running(self, spec):
        side_effect = [_completed("1;COMPLETING\n"), _completed("")]
        with patch("crane.distributed.slurm._run", side_effect=side_effect):
            view = Slurm(num_jobs=4).poll(spec, ["111", "222"])
        assert view.active_jobs == {1: JobState.RUNNING}


class TestCancel:
    def test_cancels_every_job(self, spec):
        with patch("crane.distributed.slurm.shutil.which", return_value="/usr/bin/scancel"):
            with patch("crane.distributed.slurm._run") as run:
                Slurm(num_jobs=4).cancel(spec, ["111", "222"])
        assert run.call_args.args[0] == ["scancel", "111", "222"]
