from pathlib import Path

import pytest

from benchmark_callback import BenchmarkRewardCallback, load_benchmark_sequences
from gym_lattice.envs import Lattice2DEnv


class RecordingLogger:
    def __init__(self):
        self.values = {}

    def record(self, key, value):
        self.values[key] = value


class AlwaysRightModel:
    def __init__(self):
        self.logger = RecordingLogger()

    def predict(self, observation, deterministic):
        return 3, None


class CollisionModel(AlwaysRightModel):
    def __init__(self):
        super().__init__()
        self.actions = iter((3, 0))

    def predict(self, observation, deterministic):
        return next(self.actions, 0), None


def test_load_benchmark_sequences(tmp_path: Path):
    benchmark_file = tmp_path / "benchmark.txt"
    benchmark_file.write_text("2 hh -1\n3 hph -2\n", encoding="utf-8")

    assert load_benchmark_sequences(benchmark_file) == ["HH", "HPH"]


def test_callback_logs_episode_rewards_on_interval(tmp_path: Path):
    benchmark_file = tmp_path / "benchmark.txt"
    benchmark_file.write_text("2 HH 0\n3 HPH 0\n", encoding="utf-8")
    training_env = Lattice2DEnv(5, "HHH")
    callback = BenchmarkRewardCallback(benchmark_file, training_env, eval_freq=100_000)
    model = AlwaysRightModel()
    callback.model = model
    callback.num_timesteps = 99_999

    assert callback._on_step()
    assert model.logger.values == {}

    callback.num_timesteps = 100_000
    assert callback._on_step()

    assert model.logger.values["benchmark/01_HH/total_reward"] == -0.01
    assert model.logger.values["benchmark/02_HPH/total_reward"] == -0.03
    assert training_env.seq == "HHH"
    assert callback.next_eval_timestep == 200_000


def test_callback_stops_and_marks_nonterminating_collision_episode(tmp_path: Path):
    benchmark_file = tmp_path / "benchmark.txt"
    benchmark_file.write_text("3 HHH 0\n", encoding="utf-8")
    callback = BenchmarkRewardCallback(
        benchmark_file,
        Lattice2DEnv(5, "HHH"),
        eval_freq=1,
    )
    model = CollisionModel()
    callback.model = model

    callback._evaluate_benchmarks()

    assert model.logger.values["benchmark/01_HHH/total_reward"] == pytest.approx(-2.02)
    assert model.logger.values["benchmark/01_HHH/episode_complete"] == 0.0
