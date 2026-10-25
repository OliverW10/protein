"""Sequence-view adapters for the lattice environment."""

import gymnasium as gym
import numpy as np

from Oliver.envs import (
    DistancePenalty,
    FoldMetrics,
    GuardAgainstLatticeTrap,
    count_contacts,
    get_lattice_env,
    make_env as make_grid_env,
    moves_to_coords,
    random_sequence_fn,
    valid_move_mask,
)


class SequenceObservation(gym.ObservationWrapper):
    """Expose residue type, placement action, and valid moves to an LSTM."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        lattice = get_lattice_env(env)
        self.observation_space = gym.spaces.Dict(
            {
                "sequence": gym.spaces.Box(
                    0, 1, shape=(len(lattice.seq), 7), dtype=np.float32
                ),
                "valid_moves": gym.spaces.Box(0, 1, shape=(4,), dtype=np.float32),
            }
        )

    def observation(self, _grid_observation):
        lattice = get_lattice_env(self.env)
        sequence = np.zeros((len(lattice.seq), 7), dtype=np.float32)
        for index, residue in enumerate(lattice.seq):
            if 1 <= index <= len(lattice.actions):
                sequence[index, 1 + lattice.actions[index - 1]] = 1.0
            else:
                sequence[index, 0] = 1.0
            sequence[index, 5 if residue == "P" else 6] = 1.0
        return {
            "sequence": sequence,
            "valid_moves": valid_move_mask(self.env),
        }


def make_env(
    seq_fn,
    sequence_view: bool = False,
    collision_penalty: float = -2,
    trap_penalty: float = 0.5,
    distance_penalty: float = 0.01,
    max_steps_factor: int = 4,
) -> gym.Env:
    """Build an adapted environment, optionally exposing sequence features."""
    env = make_grid_env(
        seq_fn,
        collision_penalty=collision_penalty,
        trap_penalty=trap_penalty,
        distance_penalty=distance_penalty,
        max_steps_factor=max_steps_factor,
    )
    if sequence_view:
        env = SequenceObservation(env)
    return env
