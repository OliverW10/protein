"""Sequence-view adapters for the lattice environment."""

import gymnasium as gym
import numpy as np

from Oliver.envs import (
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
        self._cached_sequence: str | None = None
        self._cached_observation: np.ndarray | None = None
        self._cached_actions: tuple[int, ...] = ()

    def _reset_sequence_cache(self, sequence: str) -> None:
        observation = np.zeros((len(sequence), 7), dtype=np.float32)
        observation[:, 0] = 1.0
        residue_channels = np.fromiter(
            (5 if residue == "P" else 6 for residue in sequence),
            dtype=np.intp,
            count=len(sequence),
        )
        observation[np.arange(len(sequence)), residue_channels] = 1.0
        self._cached_sequence = sequence
        self._cached_observation = observation
        self._cached_actions = ()

    def observation(self, _grid_observation):
        lattice = get_lattice_env(self.env)
        if self._cached_sequence != lattice.seq or self._cached_observation is None:
            self._reset_sequence_cache(lattice.seq)

        actions = tuple(lattice.actions)
        if (
            len(actions) < len(self._cached_actions)
            or actions[: len(self._cached_actions)] != self._cached_actions
        ):
            self._reset_sequence_cache(lattice.seq)

        observation = self._cached_observation
        if observation is None:
            raise RuntimeError("sequence observation cache was not initialized")
        for index, action in enumerate(
            actions[len(self._cached_actions) :],
            start=len(self._cached_actions),
        ):
            residue_index = index + 1
            observation[residue_index, :5] = 0.0
            observation[residue_index, 1 + action] = 1.0
        self._cached_actions = actions

        return {
            "sequence": observation.copy(),
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
        include_next_residues=not sequence_view,
    )
    if sequence_view:
        env = SequenceObservation(env)
    return env
