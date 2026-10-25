"""Feature extractors for fully connected policies."""

import torch.nn as nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomANN(BaseFeaturesExtractor):
    """Two-layer fully connected extractor for flattened lattice grids."""

    def __init__(self, observation_space, features_dim: int = 84, hidden_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.network = nn.Sequential(
            nn.Linear(get_flattened_obs_dim(observation_space), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.network(observations)
