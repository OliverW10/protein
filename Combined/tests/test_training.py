import pytest
import numpy as np
from gym_lattice.envs import Lattice2DEnv
from stable_baselines3 import DQN, PPO

import Combined.train as training
from Combined.train import build_model
from Muhammad.dqn.callbacks import (
    EpisodeExponentialDecay,
    TimestepExponentialDecay,
)
from Muhammad.dqn.envs import make_env as make_lstm_env
from Muhammad.dqn.networks import CustomLSTM
from Oliver.envs import (
    DistancePenalty,
    NextResiduesObservation,
    make_env as make_fc_env,
)
from Oliver.networks import CustomANN
from Willie.custom_cnn import CNN_FEATURES_DIM, CNNObservation, CustomCNN


def test_full_run_evaluation_interval_is_200_episodes():
    assert training.EVALUATION_INTERVAL == 200


def test_timestep_epsilon_decay_matches_episode_decay_progress():
    timestep_schedule = TimestepExponentialDecay(0.01, 1.0, 5)
    episode_schedule = EpisodeExponentialDecay(100, 0.01, 1.0, 5)

    for progress in (0.0, 0.25, 0.5, 1.0):
        episode_schedule.episodes = int(100 * progress)
        assert timestep_schedule(1.0 - progress) == pytest.approx(
            episode_schedule(1.0 - progress)
        )


def test_timestep_epsilon_decay_reaches_minimum_after_first_80_percent():
    schedule = TimestepExponentialDecay(0.01, 1.0, 5, decay_fraction=0.8)

    assert schedule(1.0) == pytest.approx(1.0)
    assert schedule(0.6) > 0.01
    assert schedule(0.2) == pytest.approx(0.01)
    assert schedule(0.0) == pytest.approx(0.01)


def test_fc_environment_applies_distance_penalty_outside_the_base_env():
    env = make_fc_env("HPHPH", distance_penalty=0.5)
    env.reset()

    _, reward, _, _, _ = env.step(0)

    assert reward == -0.5
    assert not hasattr(env.unwrapped, "distance_penalty")
    assert env.observation_space.contains(env.reset()[0])
    assert env.observation_space.shape[0] == (
        env.unwrapped.observation_space.shape[0] + 20 * 3
    )


def test_next_residues_wrapper_exposes_upcoming_residues_and_padding():
    base_env = Lattice2DEnv(5, "HPHPH")
    env = NextResiduesObservation(base_env, lookahead=5)

    observation, _ = env.reset()

    assert set(observation) == {"grid", "next_residues"}
    assert np.array_equal(observation["next_residues"], [2, 1, 2, 1, 0])
    assert env.observation_space.contains(observation)

    observation, _, _, _, _ = env.step(0)
    assert np.array_equal(observation["next_residues"], [1, 2, 1, 0, 0])


@pytest.mark.parametrize("lookahead", [0, -1, 1.5, True])
def test_next_residues_wrapper_rejects_invalid_lookahead(lookahead):
    with pytest.raises(ValueError, match="lookahead"):
        NextResiduesObservation(Lattice2DEnv(5, "HPHPH"), lookahead=lookahead)


def test_lstm_environment_exposes_sequence_and_valid_move_features():
    env = make_lstm_env("HPHPH", sequence_view=True)

    observation, _ = env.reset()

    assert observation["sequence"].shape == (5, 7)
    assert observation["valid_moves"].shape == (4,)
    assert env.observation_space.contains(observation)


def test_lstm_sequence_observation_updates_actions_without_mutating_prior_obs():
    env = make_lstm_env("HPHPH", sequence_view=True)
    initial_observation, _ = env.reset()
    initial_sequence = initial_observation["sequence"].copy()

    next_observation, _, _, _, _ = env.step(0)

    assert not np.shares_memory(
        initial_observation["sequence"], next_observation["sequence"]
    )
    assert np.array_equal(initial_observation["sequence"], initial_sequence)
    assert np.array_equal(next_observation["sequence"][1, :5], [0, 1, 0, 0, 0])
    assert next_observation["sequence"][1, 5] == 1

    reset_observation, _ = env.reset()
    assert np.array_equal(reset_observation["sequence"][1, :5], [1, 0, 0, 0, 0])


@pytest.mark.parametrize(
    ("algorithm", "network", "expected_model", "expected_extractor"),
    [
        ("dqn", "fc", DQN, CustomANN),
        ("dqn", "lstm", DQN, CustomLSTM),
        ("dqn", "cnn", DQN, CustomCNN),
        ("ppo", "fc", PPO, CustomANN),
        ("ppo", "lstm", PPO, CustomLSTM),
        ("ppo", "cnn", PPO, CustomCNN),
    ],
)
def test_all_model_combinations_build(
    algorithm, network, expected_model, expected_extractor
):
    env = training._make_env("HPHPH", network)

    model = build_model(
        algorithm,
        network,
        env,
        seed=7,
        device="cpu",
        total_timesteps=64,
        smoke=True,
    )
    extractor = (
        model.q_net.features_extractor
        if algorithm == "dqn"
        else model.policy.features_extractor
    )

    assert isinstance(model, expected_model)
    assert isinstance(extractor, expected_extractor)
    if network == "lstm":
        assert extractor.lstm.hidden_size == 256
        assert extractor.lstm.num_layers == 2
    elif network == "cnn":
        assert env.observation_space.shape == (5, 11, 11)
        assert extractor.features_dim == CNN_FEATURES_DIM == 128


def test_cnn_observation_encodes_fold_and_next_residue():
    env = training._make_env("HPHPH", "cnn")

    observation, _ = env.reset()

    assert isinstance(env, CNNObservation)
    assert observation.shape == (5, 11, 11)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert observation[0, 5, 5] == 1.0
    assert not observation[3].any()
    assert np.allclose(observation[4], 1 / 5)

    observation, _, _, _, _ = env.step(0)
    assert observation[3].all()


def test_combined_dqn_uses_four_gradient_steps_outside_smoke_mode():
    env = training._make_env("HPHPH", "fc")

    model = build_model(
        "dqn",
        "fc",
        env,
        seed=7,
        device="cpu",
        total_timesteps=64,
    )

    assert model.gradient_steps == 4


@pytest.mark.parametrize("smoke", [True, False])
def test_combined_ppo_uses_1024_step_rollouts(smoke):
    env = training._make_env("HPHPH", "fc")

    model = build_model(
        "ppo",
        "fc",
        env,
        seed=7,
        device="cpu",
        total_timesteps=64,
        smoke=smoke,
    )

    assert model.n_steps == 1024


def test_main_appends_benchmark_summary_without_duplicate_header(tmp_path, monkeypatch):
    summary_path = tmp_path / "benchmark_summary.csv"
    summary_path.write_text("algorithm,network,seed\nppo,lstm,1\n", encoding="utf-8")
    monkeypatch.setattr(training, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(training, "NETWORKS", ["fc"])
    monkeypatch.setattr(
        training,
        "train_run",
        lambda *args, **kwargs: {
            "algorithm": "ppo",
            "network": "fc",
            "seed": 7,
        },
    )

    training.main()

    assert summary_path.read_text(encoding="utf-8").splitlines() == [
        "algorithm,network,seed",
        "ppo,lstm,1",
        "ppo,fc,7",
    ]
