import argparse
import os
import random

import gymnasium as gym
from stable_baselines3 import DQN

from .evaluate import watch_agent

# Each agent's best model on the test proteins.
demo_models = {
    "lstm": "logs/combined/lstm-dqn/seed7/best_model.zip",
    "fc": "dqn/models/fc_200k_seed21_best.zip",
}


# Fold one 20-residue H/P protein with a saved model and show the env's terminal picture after every move
def demo(seq, net):

    model_path = demo_models[net]
    model = DQN.load(model_path, device="cpu")
    print("Protein:", seq)
    print("Model:", model_path)
    watch_agent(model, net, seq)


# Run from the repository root like: python -m Muhammad.dqn.demo HPHPPHHPHPPHPHHPPHPH --net lstm
def main():

    parser = argparse.ArgumentParser(description="Fold a 20-residue H/P protein with a model")
    parser.add_argument("seq", nargs="?", default=None, help="20 letters of H and P (a random protein if left out)")
    parser.add_argument("--net", choices=["fc", "lstm"], default="lstm")
    args = parser.parse_args()

    # Makes older consoles show the env's colours
    gym.logger.min_level = gym.logger.ERROR

    if args.seq is None:
        seq = "".join(random.choice("HP") for i in range(20))
    else:
        seq = args.seq.upper()
    if len(seq) != 20 or set(seq) - set("HP"):
        raise SystemExit("The sequence must be exactly 20 letters of H and P")

    demo(seq, args.net)


if __name__ == "__main__":
    main()
