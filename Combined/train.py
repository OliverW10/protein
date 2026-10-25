"""Train and compare FC/LSTM feature extractors with DQN and PPO."""

import csv
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import TypedDict

import gymnasium as gym
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from Muhammad.dqn.callbacks import (
    GreedyEvalCallback,
    SaveOnIntervalCallback,
    greedy_episode,
)
from Muhammad.dqn.envs import make_env, random_sequence_fn
from Muhammad.dqn.networks import CustomLSTM
from Muhammad.dqn.train import (
    DQNParameters,
    checkpoint_every,
    dqn_parameters,
    eps_decay,
    eps_max,
    eps_min,
    max_steps_factor,
    num_test,
    num_validation,
    seq_length,
    sequences,
    test_seed,
    validation_seed,
)
from Oliver.networks import CustomANN


SMOKE = False
NETWORKS = ["fc", "lstm"]
ALGORITHMS = ["dqn", "ppo"]
SEEDS = [7] if SMOKE else [7, 14, 21]
TOTAL_TIMESTEPS = 256 if SMOKE else 100_000
NUM_VALIDATION_SEQUENCES = 2 if SMOKE else num_validation
EVALUATION_INTERVAL = 1 if SMOKE else 1000
NUM_TEST_SEQUENCES = 2 if SMOKE else num_test
DEVICE = "auto"
THREADS = 1
OUTPUT_DIR = Path("logs/combined")


class ExponentialTimestepDecay:
    """Decay DQN exploration according to normalized training progress."""

    def __init__(self, eps_min: float, eps_max: float, eps_decay: float):
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay

    def __call__(self, progress_remaining: float) -> float:
        progress = min(max(1.0 - progress_remaining, 0.0), 1.0)
        decay = math.exp(-self.eps_decay * progress)
        return self.eps_min + (self.eps_max - self.eps_min) * decay


class StopAtTimesteps(BaseCallback):
    """Stop learning once the requested number of environment steps is reached."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        return self.num_timesteps < self.total_timesteps


def _policy_settings(net: str):
    if net == "fc":
        return "MlpPolicy", CustomANN
    if net == "lstm":
        return "MultiInputPolicy", CustomLSTM
    raise ValueError(f"unsupported network: {net}")


def build_model(
    algorithm: str,
    net: str,
    env: gym.Env,
    seed: int,
    device: str,
    total_timesteps: int,
    smoke: bool = False,
    tensorboard_log: str | None = None,
) -> DQN | PPO:
    """Construct one model using the matching FC or LSTM feature extractor."""
    policy, extractor = _policy_settings(net)
    policy_kwargs = {"features_extractor_class": extractor, "net_arch": []}

    if algorithm == "dqn":
        dqn_kwargs: DQNParameters = dqn_parameters(seq_length, total_timesteps)
        if smoke:
            dqn_kwargs["buffer_size"] = 256
            dqn_kwargs["learning_starts"] = 4
            dqn_kwargs["batch_size"] = 8
            dqn_kwargs["gradient_steps"] = 1
        return DQN(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            **dqn_kwargs,
        )
    if algorithm == "ppo":
        if smoke:
            return PPO(
                policy,
                env,
                policy_kwargs=policy_kwargs,
                seed=seed,
                device=device,
                tensorboard_log=tensorboard_log,
                learning_rate=1e-4,
                gamma=0.98,
                ent_coef=0.001,
                n_steps=32,
                batch_size=8,
                n_epochs=1,
            )
        return PPO(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            learning_rate=1e-4,
            gamma=0.98,
            ent_coef=0.001,
        )
    raise ValueError(f"unsupported algorithm: {algorithm}")


def _make_validation_envs(net: str, count: int):
    next_validation = random_sequence_fn(seq_length, validation_seed)
    return [
        make_env(next_validation(), sequence_view=(net == "lstm"))
        for _ in range(count)
    ]


def _test_sequences(count: int) -> OrderedDict[str, str]:
    result: OrderedDict[str, str] = OrderedDict(
        (name, entry["seq"]) for name, entry in sequences.items()
    )
    next_test = random_sequence_fn(seq_length, test_seed)
    for index in range(count):
        result[f"test-{index:03d}"] = next_test()
    return result


def evaluate_model(model, net: str, test_count: int, output_dir: Path) -> dict:
    """Evaluate greedily on fixed and seeded test sequences; save fold metrics."""
    rows = []
    for name, sequence in _test_sequences(test_count).items():
        env = make_env(sequence, sequence_view=(net == "lstm"))
        result = greedy_episode(model, env)
        rows.append(
            {
                "name": name,
                "sequence": sequence,
                "contacts": result["contacts"],
                "complete": result["complete"],
                "energy": -result["contacts"] if result["complete"] else "",
            }
        )

    results_path = output_dir / "test_folds.csv"
    with results_path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    random_rows = [row for row in rows if row["name"].startswith("test-")]
    summary = {
        "test_sequences": len(random_rows),
        "mean_contacts": (
            sum(row["contacts"] if row["complete"] else 0 for row in random_rows)
            / len(random_rows)
            if random_rows
            else 0.0
        ),
        "completion_rate": (
            sum(row["complete"] for row in random_rows) / len(random_rows)
            if random_rows
            else 0.0
        ),
    }
    summary_path = output_dir / "test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_run(
    algorithm: str,
    net: str,
    seed: int,
    total_timesteps: int,
    num_validation_sequences: int,
    eval_every: int,
    num_test_sequences: int,
    device: str,
    threads: int,
    output_root: Path,
    smoke: bool = False,
) -> dict:
    """Train one architecture/algorithm/seed run and save its evaluation."""
    torch.set_num_threads(threads)
    gym.logger.min_level = gym.logger.ERROR

    run_dir = output_root / f"{net}-{algorithm}" / f"seed{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(
        random_sequence_fn(seq_length, seed),
        sequence_view=(net == "lstm"),
        max_steps_factor=max_steps_factor,
    )
    validation_envs = _make_validation_envs(net, num_validation_sequences)
    model = build_model(
        algorithm,
        net,
        env,
        seed,
        device,
        total_timesteps,
        smoke=smoke,
        tensorboard_log=str(run_dir),
    )
    callbacks = [
        GreedyEvalCallback(validation_envs, str(run_dir), eval_every),
        SaveOnIntervalCallback(checkpoint_every, str(checkpoint_dir)),
    ]
    if isinstance(model, DQN):
        model.exploration_schedule = ExponentialTimestepDecay(
            eps_min, eps_max, eps_decay
        )
    callbacks.append(StopAtTimesteps(total_timesteps))

    print(
        f"Training {net.upper()}-{algorithm.upper()} seed {seed}: "
        f"{total_timesteps:,} timesteps on {model.device}"
    )
    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=100,
        tb_log_name=f"{net}-{algorithm}-seed{seed}",
    )
    training_minutes = round((time.time() - start) / 60, 1)
    model.save(str(run_dir / "final_model"))
    best_model_path = run_dir / "best_model.zip"
    evaluation_model = (
        type(model).load(str(best_model_path), device=device)
        if best_model_path.exists()
        else model
    )
    test_summary = evaluate_model(
        evaluation_model, net, num_test_sequences, run_dir
    )
    if isinstance(model, DQN):
        extractor = model.q_net.features_extractor
    else:
        extractor = model.policy.features_extractor
    design = {
        "algorithm": algorithm,
        "network": net,
        "seed": seed,
        "target_timesteps": total_timesteps,
        "timesteps": model.num_timesteps,
        "training_minutes": training_minutes,
        "feature_extractor_parameters": sum(
            parameter.numel() for parameter in extractor.parameters()
        ),
        **test_summary,
    }
    (run_dir / "design_settings.json").write_text(
        json.dumps(design, indent=2), encoding="utf-8"
    )
    print(
        f"Finished {net.upper()}-{algorithm.upper()} seed {seed}: "
        f"mean contacts {test_summary['mean_contacts']:.2f}, "
        f"completion {test_summary['completion_rate']:.2%}"
    )
    return design


def main() -> None:
    if (
        TOTAL_TIMESTEPS < 1
        or NUM_VALIDATION_SEQUENCES < 1
        or EVALUATION_INTERVAL < 1
        or NUM_TEST_SEQUENCES < 0
    ):
        raise ValueError(
            "timesteps and evaluation counts must be positive "
            "(test count may be zero)"
        )

    results = []
    for net in NETWORKS:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                results.append(
                    train_run(
                        algorithm,
                        net,
                        seed,
                        TOTAL_TIMESTEPS,
                        NUM_VALIDATION_SEQUENCES,
                        EVALUATION_INTERVAL,
                        NUM_TEST_SEQUENCES,
                        DEVICE,
                        THREADS,
                        OUTPUT_DIR,
                        smoke=SMOKE,
                    )
                )

    if results:
        summary_path = OUTPUT_DIR / "benchmark_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved matrix results to {summary_path}")


if __name__ == "__main__":
    main()
