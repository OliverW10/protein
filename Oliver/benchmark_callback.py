"""Periodic benchmark evaluation callback for Stable-Baselines3."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


def load_benchmark_sequences(path: str | Path) -> list[str]:
    """Load sequences from whitespace-separated length/sequence/score rows."""
    benchmark_path = Path(path)
    sequences: list[str] = []
    with benchmark_path.open(encoding="utf-8") as benchmark_file:
        for line_number, line in enumerate(benchmark_file, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(
                    f"{benchmark_path}:{line_number}: expected a length and sequence"
                )
            try:
                declared_length = int(fields[0])
            except ValueError as error:
                raise ValueError(
                    f"{benchmark_path}:{line_number}: invalid sequence length {fields[0]!r}"
                ) from error

            sequence = fields[1].upper()
            if (
                declared_length < 1
                or declared_length != len(sequence)
                or not set(sequence) <= {"H", "P"}
            ):
                raise ValueError(
                    f"{benchmark_path}:{line_number}: sequence length or residues are invalid"
                )
            sequences.append(sequence)

    if not sequences:
        raise ValueError(f"{benchmark_path}: no benchmark sequences found")
    return sequences


class BenchmarkRewardCallback(BaseCallback):
    """Evaluate fixed benchmark sequences periodically and log episode rewards."""

    def __init__(
        self,
        benchmark_path: str | Path,
        eval_env: Any,
        eval_freq: int = 100_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if isinstance(eval_freq, bool) or not isinstance(eval_freq, int) or eval_freq < 1:
            raise ValueError("eval_freq must be a positive integer")
        self.sequences = load_benchmark_sequences(benchmark_path)
        self.eval_env = deepcopy(eval_env)
        self.eval_freq = eval_freq
        self.next_eval_timestep = eval_freq

    def _on_training_start(self) -> None:
        self.next_eval_timestep = (
            self.num_timesteps // self.eval_freq + 1
        ) * self.eval_freq

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_eval_timestep:
            self._evaluate_benchmarks()
            self.next_eval_timestep += self.eval_freq
        return True

    def _evaluate_benchmarks(self) -> None:
        for index, sequence in enumerate(self.sequences, start=1):
            self.eval_env.seq_fn = lambda sequence=sequence: sequence
            observation, _ = self.eval_env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = self.model.predict(observation, deterministic=True)
                chain_length = len(self.eval_env.state)
                observation, reward, terminated, truncated, _ = self.eval_env.step(
                    int(action)
                )
                total_reward += float(reward)
                if len(self.eval_env.state) == chain_length and not terminated:
                    break

            metric_prefix = f"benchmark/{index:02d}_{sequence}"
            self.logger.record(f"{metric_prefix}/total_reward", total_reward)
            self.logger.record(
                f"{metric_prefix}/episode_complete",
                float(terminated or truncated),
            )
