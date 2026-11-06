import argparse
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from Oliver.envs import make_env as make_grid_env
from Willie.custom_cnn import CNNObservation
from .envs import make_env
from .evaluate import render_chain
from .networks import CustomANN, CustomLSTM

# Model files are found from the repository root so the demo runs from any folder
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Each agent's best model on the test proteins (FC 200K seed 21 best / LSTM 200K seed 21 final)
# The CNN models were each trained on one benchmark only (cnn-a on 20mer-A and cnn-b on 20mer-B)
demo_models = {
    "lstm": os.path.join(root, "Muhammad", "dqn", "models", "lstm_200k_seed21_final.zip"),
    "fc": os.path.join(root, "Muhammad", "dqn", "models", "fc_200k_seed21_best.zip"),
    "cnn-a": os.path.join(root, "Willie", "20mer-A_model_500000.zip"),
    "cnn-b": os.path.join(root, "Willie", "20mer-B_model_500000.zip"),
}

# The benchmark each CNN model was trained on is folded when no protein is given
cnn_sequences = {
    "cnn-a": "HPHPPHHPHPPHPHHPPHPH",
    "cnn-b": "HHHPPHPHPHPPHPHPHPPH",
}


class SavedCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=256):

        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):

        return self.linear(self.cnn(observations))


class NextFiveResidues(gym.ObservationWrapper):

    def __init__(self, env):

        super().__init__(env)
        size = env.observation_space.shape[0] + 5 * 3
        self.observation_space = gym.spaces.Box(0, 1, shape=(size,), dtype=np.int64)

    def observation(self, grid_obs):

        lattice = self.env.unwrapped
        placed = len(lattice.state)
        upcoming = lattice.seq[placed:placed + 5].ljust(5, "_")

        preview = np.zeros(5 * 3, dtype=np.int64)
        for i, residue in enumerate(upcoming):
            preview[3 * i + "_HP".index(residue)] = 1
        return np.concatenate([grid_obs, preview])


class SwapUpDown(gym.ActionWrapper):

    def action(self, action):

        swap = {0: 0, 1: 2, 2: 1, 3: 3}
        return swap[int(action)]


def load_model(net):

    if net == "fc":
        policy_kwargs = {"features_extractor_class": CustomANN,
                         "features_extractor_kwargs": {"features_dim": 84, "hidden_dim": 256}, "net_arch": []}
    elif net == "lstm":
        policy_kwargs = {"features_extractor_class": CustomLSTM,
                         "features_extractor_kwargs": {"hidden_dim": 256, "num_layers": 2}, "net_arch": []}
    else:
        policy_kwargs = {"features_extractor_class": SavedCNN, "features_extractor_kwargs": {"features_dim": 256},
                         "net_arch": [256], "normalize_images": False}

    custom_objects = {
        "policy_kwargs": policy_kwargs,
        "lr_schedule": lambda _: 0.0,
        "exploration_schedule": lambda _: 0.0,
    }
    return DQN.load(demo_models[net], device="cpu", custom_objects=custom_objects)


def build_env(net, seq):

    if net == "lstm":
        return make_env(seq, sequence_view=True)
    if net == "fc":
        return SwapUpDown(NextFiveResidues(make_grid_env(seq, include_next_residues=False)))
    return SwapUpDown(CNNObservation(make_grid_env(seq, include_next_residues=False)))


def watch(model, env):

    obs, _ = env.reset()
    render_chain(env)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        render_chain(env)
        if terminated or truncated:
            break

    print("Complete:", info["complete"], "| Contacts:", info["contacts"], "| Collisions:", info["collisions"],
          "| Moves:", "".join(info["actions"]))


# Fold one 20-residue H/P protein with a saved model
def demo(seq, net):

    model = load_model(net)
    print("Protein:", seq)
    print("Model:", os.path.relpath(demo_models[net], root))
    if net in cnn_sequences and seq != cnn_sequences[net]:
        print("Note: this CNN model was only trained on", cnn_sequences[net])
    watch(model, build_env(net, seq))


# Run from the repository root like: python -m Muhammad.dqn.demo HPHPPHHPHPPHPHHPPHPH --net lstm (or --net fc / cnn-a / cnn-b)
def main():

    parser = argparse.ArgumentParser(description="Fold a 20-residue H/P protein with a saved model")
    parser.add_argument("seq", nargs="?", default=None, help="20 letters of H and P (a random protein if left out or the CNN's own benchmark)")
    parser.add_argument("--net", choices=["fc", "lstm", "cnn-a", "cnn-b"], default="lstm")
    args = parser.parse_args()

    # Turns on the colour codes the env's terminal picture uses in consoles that don't show them by default
    os.system("")
    gym.logger.min_level = gym.logger.ERROR

    if args.seq is None and args.net in cnn_sequences:
        seq = cnn_sequences[args.net]
    elif args.seq is None:
        seq = "".join(random.choice("HP") for i in range(20))
    else:
        seq = args.seq.upper()
    if len(seq) != 20 or set(seq) - set("HP"):
        raise SystemExit("The sequence must be exactly 20 letters of H and P")

    demo(seq, args.net)


if __name__ == "__main__":
    main()
