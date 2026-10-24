# Settings adapted from Yang, K., Huang, H., Vandans, O., Murali, A., Tian, F., Yap, R. H. C., & Dai, L. (2023). Applying deep reinforcement learning to the HP model for protein structure prediction. Physica A: Statistical Mechanics and Its Applications, 609, 128395. https://doi.org/10.1016/j.physa.2022.128395
import argparse
import json
import os
import time
from collections import OrderedDict

import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

from dqn.callbacks import EpisodeEpsilonCallback, EpisodeExponentialDecay, GreedyEvalCallback, SaveOnIntervalCallback
from dqn.envs import make_env, random_sequence_fn
from dqn.networks import CustomANN, CustomLSTM

# Benchmark sequences and their best known energies (energy = -number of H-H contacts so lower is better)
sequences = OrderedDict([
    ("20mer-A", {"seq": "HPHPPHHPHPPHPHHPPHPH", "best_energy": -9}),
    ("20mer-B", {"seq": "HHHPPHPHPHPPHPHPHPPH", "best_energy": -10}),
])

# Every run trains on random sequences of this length
seq_length = 20

# Number of training episodes (the longer runs pass --episodes 200000)
episodes = 100_000

# Test different seeds
seeds = [7, 14, 21]

# Random proteins for picking the best model (validation) and for the final results (test)
validation_seed = 1000
test_seed = 2000
num_validation = 50
num_test = 100

# Check the validation proteins every eval_every episodes and save a checkpoint every checkpoint_every steps
eval_every = 1000
checkpoint_every = 500_000

# Environment settings
env_parameters = OrderedDict([
    ("collision_penalty", -2),
    ("trap_penalty", 0.5),
    ("distance_penalty", 0.01),
])

# Cut episodes off after max_steps_factor * N steps since collisions can repeat forever
max_steps_factor = 4

# Epsilon decays each episode: eps = eps_min + (eps_max - eps_min) * exp(-eps_decay * episode / total_episodes)
eps_max = 1.0
eps_min = 0.01
eps_decay = 5

log_root = "logs/dqn"


# DQN settings written as Stable-Baselines3 arguments
def dqn_parameters(seq_length, num_episodes):
    config = OrderedDict([
        ("learning_rate", 5e-4),
        ("batch_size", 32),
        ("gamma", 0.98),
        ("buffer_size", max(min(50_000, num_episodes // 10), 50 * seq_length)),
        # Act randomly for the first 50 * N steps to fill the buffer before learning
        ("learning_starts", 50 * seq_length),
        # 10 gradient steps at the end of every episode
        ("train_freq", (1, "episode")),
        ("gradient_steps", 10),
        # Update the target network about every 100 episodes
        ("target_update_interval", 100 * (seq_length - 1)),
    ])
    return config


# Make the DQN with the chosen network
def build_model(net, env, seq_length, num_episodes, seed, device, log_dir):

    # fc reads the flat grid observation and lstm reads the sequence view (a dict)
    if net == "fc":
        policy = "MlpPolicy"
        extractor = CustomANN
    else:
        policy = "MultiInputPolicy"
        extractor = CustomLSTM

    # net_arch=[] so SB3 only adds one layer from the features to the 4 Q-values
    policy_kwargs = {"features_extractor_class": extractor, "net_arch": []}
    model = DQN(policy, env, policy_kwargs=policy_kwargs, seed=seed, device=device,
                tensorboard_log=log_dir, **dqn_parameters(seq_length, num_episodes))

    return model


# Train one DQN on random sequences of one length and save everything in its own folder
def train(net, seed=7, num_episodes=None, device="auto", tag="", threads=1):

    if num_episodes is None:
        num_episodes = episodes

    # One CPU thread each so several runs can share the CPU
    torch.set_num_threads(threads)
    # Hide the env's "agent was trapped" warning
    gym.logger.min_level = gym.logger.ERROR

    run_dir = os.path.join(log_root, f"{seq_length}mer", net + tag, f"seed{seed}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    sequence_view = net == "lstm"

    # Training proteins come from the run's seed and the validation proteins from their own seed
    env = make_env(random_sequence_fn(seq_length, seed), sequence_view, max_steps_factor=max_steps_factor, **env_parameters)
    next_validation = random_sequence_fn(seq_length, validation_seed)
    validation_envs = []
    for i in range(num_validation):
        validation_envs.append(make_env(next_validation(), sequence_view, max_steps_factor=max_steps_factor, **env_parameters))

    model = build_model(net, env, seq_length, num_episodes, seed, device, run_dir)
    schedule = EpisodeExponentialDecay(num_episodes, eps_min, eps_max, eps_decay)
    callbacks = [
        EpisodeEpsilonCallback(schedule),
        GreedyEvalCallback(validation_envs, run_dir, eval_every),
        SaveOnIntervalCallback(checkpoint_every, os.path.join(run_dir, "checkpoints")),
        StopTrainingOnMaxEpisodes(num_episodes),
    ]

    print(f"Training {net} on random {seq_length}-mers (seed {seed}) for {num_episodes} episodes on {model.device}")
    start = time.time()

    # Training stops after num_episodes so this many steps is never reached
    model.learn(total_timesteps=num_episodes * max_steps_factor * seq_length, callback=callbacks, log_interval=100)
    model.save(os.path.join(run_dir, "final_model"))

    # The run details shown in the notebook's run table
    design = {
        "seed": seed,
        "num_episodes": num_episodes,
        "timesteps": model.num_timesteps,
        "training_minutes": round((time.time() - start) / 60, 1),
        "q_network_parameters": sum(p.numel() for p in model.q_net.parameters()),
    }
    with open(os.path.join(run_dir, "design_settings.json"), "w") as f:
        json.dump(design, f, indent=4)

    print("Done in", design["training_minutes"], "minutes and saved to", run_dir)
    return run_dir


# Run from the repo folder like: python -m dqn.train --net fc --seed 7
def main():

    parser = argparse.ArgumentParser(description="Train a DQN on random HP sequences")
    parser.add_argument("--net", choices=["fc", "lstm"], default="fc")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    train(args.net, args.seed, args.episodes, args.device, args.tag, args.threads)


if __name__ == "__main__":
    main()
