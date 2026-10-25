import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from Oliver.networks import CustomANN


# LSTM that reads the sequence view one residue at a time
class CustomLSTM(BaseFeaturesExtractor):

    def __init__(self, observation_space, hidden_dim=256, num_layers=2):

        # The LSTM output plus the 4 numbers from valid_moves
        features_dim = hidden_dim + observation_space["valid_moves"].shape[0]
        super().__init__(observation_space, features_dim)

        num_columns = observation_space["sequence"].shape[1]
        self.lstm = nn.LSTM(num_columns, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, observations):

        # Starts fresh every step since the sequence view already holds the whole history
        output, _ = self.lstm(observations["sequence"])

        # Keep the output after the last residue and add the free moves to it
        last_output = output[:, -1]
        return torch.cat([last_output, observations["valid_moves"]], dim=1)
