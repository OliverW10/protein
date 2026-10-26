"""Environment adapters shared by the fully connected experiments."""

import math
import random
from collections.abc import Callable, Sequence
from math import floor
from numbers import Integral, Real

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

from gym_lattice.envs import Lattice2DEnv


Coordinate = tuple[int, int]
RESIDUE_TO_OBSERVATION_INT = {"_": 0, "H": 1, "P": 2}


def get_lattice_env(env: gym.Env) -> Lattice2DEnv:
    """Return the underlying lattice environment with its concrete API."""
    lattice = env.unwrapped
    if not isinstance(lattice, Lattice2DEnv):
        raise TypeError("environment must wrap a Lattice2DEnv")
    return lattice


def count_contacts(coords: Sequence[Coordinate], seq: str) -> int:
    """Count non-bonded H-H lattice contacts."""
    index_at = {position: index for index, position in enumerate(coords)}
    contacts = 0
    for index, (x, y) in enumerate(coords):
        if seq[index] != "H":
            continue
        for neighbour in ((x + 1, y), (x, y + 1)):
            other = index_at.get(neighbour)
            if other is not None and seq[other] == "H" and abs(index - other) > 1:
                contacts += 1
    return contacts


def moves_to_coords(fold: str) -> list[Coordinate]:
    """Convert a string of move labels to lattice coordinates."""
    steps = {"L": (-1, 0), "D": (0, -1), "U": (0, 1), "R": (1, 0)}
    x, y = 0, 0
    coords = [(x, y)]
    for letter in fold:
        dx, dy = steps[letter]
        x += dx
        y += dy
        coords.append((x, y))
    return coords


def valid_move_mask(env: gym.Env) -> np.ndarray:
    """Return a four-element mask indicating which moves avoid collisions."""
    lattice = get_lattice_env(env)
    chain_end = next(reversed(lattice.state))
    neighbours = lattice._get_adjacent_coords(chain_end)
    return np.asarray(
        [float(neighbours[action] not in lattice.state) for action in range(4)],
        dtype=np.float32,
    )


class GuardAgainstLatticeTrap(gym.Wrapper):
    """Undo the base trap penalty when the chain was completed successfully."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        lattice = get_lattice_env(self.env)
        if info["is_trapped"] and len(lattice.state) == len(lattice.seq):
            reward = float(reward) + floor(len(lattice.seq) * lattice.trap_penalty)
            info["is_trapped"] = False
        return observation, reward, terminated, truncated, info


class DistancePenalty(gym.Wrapper):
    """Apply Manhattan-distance reward shaping outside the base environment."""

    def __init__(self, env: gym.Env, distance_penalty: float = 0.01):
        if (
            isinstance(distance_penalty, bool)
            or not isinstance(distance_penalty, Real)
            or not math.isfinite(distance_penalty)
            or distance_penalty < 0
        ):
            raise ValueError("distance_penalty must be a finite non-negative number")
        super().__init__(env)
        self.distance_penalty = float(distance_penalty)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        x, y = next(reversed(get_lattice_env(self.env).state))
        reward = float(reward) - self.distance_penalty * (abs(x) + abs(y))
        return observation, reward, terminated, truncated, info


class FoldMetrics(gym.Wrapper):
    """Add final contact count and completion status to terminal info."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            lattice = get_lattice_env(self.env)
            coords = list(lattice.state)
            info["contacts"] = count_contacts(coords, lattice.seq)
            info["complete"] = len(coords) == len(lattice.seq)
        return observation, reward, terminated, truncated, info


class NextResiduesObservation(gym.ObservationWrapper):
    """Add a one-hot lookahead of upcoming residues to the grid observation."""

    def __init__(self, env: gym.Env, lookahead: int = 20):
        if (
            isinstance(lookahead, bool)
            or not isinstance(lookahead, Integral)
            or lookahead < 1
        ):
            raise ValueError("lookahead must be a positive integer")
        super().__init__(env)
        self.lookahead = int(lookahead)
        lattice = get_lattice_env(env)
        self.observation_space = gym.spaces.Dict(
            {
                "grid": lattice.observation_space,
                "next_residues": gym.spaces.MultiDiscrete(
                    np.full((self.lookahead,), 3, dtype=np.int64)
                ),
            }
        )

    def observation(self, grid_observation: np.ndarray) -> dict[str, np.ndarray]:
        lattice = get_lattice_env(self.env)
        next_idx = len(lattice.state)
        upcoming = lattice.seq[next_idx : next_idx + self.lookahead].ljust(
            self.lookahead, "_"
        )
        next_residues = np.asarray(
            [RESIDUE_TO_OBSERVATION_INT[residue] for residue in upcoming],
            dtype=np.int64,
        )
        return {"grid": grid_observation, "next_residues": next_residues}


def random_sequence_fn(length: int, seed: int) -> Callable[[], str]:
    """Return a reproducible generator of random H/P sequences."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive integer")
    rng = random.Random(seed)

    def next_sequence() -> str:
        return "".join(rng.choices("HP", k=length))

    return next_sequence


def make_env(
    seq_fn: Callable[[], str] | str,
    collision_penalty: float = -2,
    trap_penalty: float = 0.5,
    distance_penalty: float = 0.01,
    max_steps_factor: int = 4,
    include_next_residues: bool = True,
    lookahead: int = 20,
) -> gym.Env:
    """Build a grid-observation environment with project reward adapters."""
    if isinstance(seq_fn, str):
        first_sequence = seq_fn
        sequence_factory = lambda: seq_fn
    elif callable(seq_fn):
        first_sequence = seq_fn()
        first_call = True

        def sequence_factory():
            nonlocal first_call
            if first_call:
                first_call = False
                return first_sequence
            return seq_fn()

    else:
        raise TypeError("seq_fn must be a sequence string or a callable")

    if not first_sequence:
        raise ValueError("sequence must contain at least one residue")
    env = Lattice2DEnv(
        len(first_sequence),
        sequence_factory,
        collision_penalty=collision_penalty,
        trap_penalty=trap_penalty,
    )
    env = GuardAgainstLatticeTrap(env)
    env = DistancePenalty(env, distance_penalty)
    env = TimeLimit(env, max_episode_steps=max_steps_factor * len(first_sequence))
    env = FoldMetrics(env)
    if include_next_residues:
        env = gym.wrappers.FlattenObservation(NextResiduesObservation(env, lookahead))
    return env
