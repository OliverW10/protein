"""CNN observation adapter and feature extractor for lattice folding."""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from Oliver.envs import get_lattice_env


class CNNObservation(gym.ObservationWrapper):
    """Encode the current fold as five image channels for the CNN policy."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n = len(get_lattice_env(env).seq)
        self.size = 2 * self.n + 1
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5, self.size, self.size),
            dtype=np.float32,
        )

    def observation(self, _grid_observation: np.ndarray) -> np.ndarray:
        lattice = get_lattice_env(self.env)
        chain = lattice.state
        head_x, head_y = next(reversed(chain))
        center = self.n
        observation = np.zeros(
            (5, self.size, self.size), dtype=np.float32
        )

        for index, ((x, y), residue) in enumerate(chain.items()):
            row = center - (y - head_y)
            col = center + (x - head_x)
            observation[0 if residue == "H" else 1, row, col] = 1.0
            if self.n > 1:
                observation[2, row, col] = index / (self.n - 1)

        placed = len(chain)
        if placed < self.n:
            observation[3, :, :] = float(lattice.seq[placed] == "H")
        observation[4, :, :] = placed / self.n
        return observation


class CustomCNN(BaseFeaturesExtractor):
    """Convolutional feature extractor for five-channel lattice observations."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        if len(observation_space.shape) != 3:
            raise ValueError("CNN observations must have channels, height, and width")
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            flattened_dim = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(flattened_dim, features_dim), nn.ReLU())

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
