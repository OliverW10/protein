import pytest
from stable_baselines3 import DQN, PPO

from Combined.train import build_model
from Muhammad.dqn.envs import make_env as make_lstm_env
from Muhammad.dqn.networks import CustomLSTM
from Oliver.envs import DistancePenalty, make_env as make_fc_env
from Oliver.networks import CustomANN


def test_fc_environment_applies_distance_penalty_outside_the_base_env():
    env = make_fc_env("HPHPH", distance_penalty=0.5)
    env.reset()

    _, reward, _, _, _ = env.step(0)

    assert isinstance(env.env.env, DistancePenalty)
    assert reward == -0.5
    assert not hasattr(env.unwrapped, "distance_penalty")
    assert env.observation_space.contains(env.reset()[0])


def test_lstm_environment_exposes_sequence_and_valid_move_features():
    env = make_lstm_env("HPHPH", sequence_view=True)

    observation, _ = env.reset()

    assert observation["sequence"].shape == (5, 7)
    assert observation["valid_moves"].shape == (4,)
    assert env.observation_space.contains(observation)


@pytest.mark.parametrize(
    ("algorithm", "network", "expected_model", "expected_extractor"),
    [
        ("dqn", "fc", DQN, CustomANN),
        ("dqn", "lstm", DQN, CustomLSTM),
        ("ppo", "fc", PPO, CustomANN),
        ("ppo", "lstm", PPO, CustomLSTM),
    ],
)
def test_all_model_combinations_build(
    algorithm, network, expected_model, expected_extractor
):
    env = (
        make_lstm_env("HPHPH", sequence_view=network == "lstm")
        if network == "lstm"
        else make_fc_env("HPHPH")
    )

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
