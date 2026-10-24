import random
from math import floor

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from gym_lattice.envs import Lattice2DEnv


# Count H-H contacts (two H's next to each other on the grid that aren't bonded)
def count_contacts(coords, seq):
    
    # Which residue is at each position
    index_at = {}
    for i, position in enumerate(coords):
        index_at[position] = i

    contacts = 0
    for i, (x, y) in enumerate(coords):
        if seq[i] != "H":
            continue
        
        # Only look right and up so each pair is counted once
        for neighbour in [(x + 1, y), (x, y + 1)]:
            j = index_at.get(neighbour)
            if j is not None and seq[j] == "H" and abs(i - j) > 1:
                contacts += 1
    
    return contacts


# Convert a fold like 'ULLD' into grid positions starting from (0, 0)
def moves_to_coords(fold):
    steps = {"L": (-1, 0), "D": (0, -1), "U": (0, 1), "R": (1, 0)}
    x, y = 0, 0
    coords = [(x, y)]
    for letter in fold:
        dx, dy = steps[letter]
        x += dx
        y += dy
        coords.append((x, y))
    return coords


# 1 if a move (L/D/U/R) lands on an empty cell and 0 if it would hit the chain
def valid_move_mask(env):
    lattice = env.unwrapped
    chain_end = list(lattice.state.keys())[-1]
    neighbours = lattice._get_adjacent_coords(chain_end)

    mask = np.zeros(4, dtype=np.float32)
    for action in range(4):
        
        if neighbours[action] not in lattice.state:
            mask[action] = 1.0
    
    return mask


# Lattice2DEnv penalises finished folds if the last residue is boxed in so this undoes that
class GuardAgainstLatticeTrap(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lattice = self.env.unwrapped
        complete = len(lattice.state) == len(lattice.seq)

        # A finished chain isn't really trapped so give the penalty back
        if info["is_trapped"] and complete:
            reward += floor(len(lattice.seq) * lattice.trap_penalty)
            info["is_trapped"] = False
        return obs, reward, terminated, truncated, info


# Record how the fold turned out at the end of each episode
class FoldMetrics(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            lattice = self.env.unwrapped
            coords = list(lattice.state.keys())
            info["contacts"] = count_contacts(coords, lattice.seq)
            info["complete"] = len(coords) == len(lattice.seq)
        return obs, reward, terminated, truncated, info


# Show the agent the chain one residue per row instead of the env's grid picture
class SequenceObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = len(env.unwrapped.seq)
        self.observation_space = gym.spaces.Dict({
            "sequence": gym.spaces.Box(0, 1, shape=(n, 7), dtype=np.float32),
            "valid_moves": gym.spaces.Box(0, 1, shape=(4,), dtype=np.float32),
        })

    def observation(self, grid_obs):
        lattice = self.env.unwrapped
        # Columns: no move/L/D/U/R for how the residue was placed then P/H for its type
        sequence = np.zeros((len(lattice.seq), 7), dtype=np.float32)
        for i, residue in enumerate(lattice.seq):
            # Residue i was placed by move i - 1 and the first residue never moves
            if 1 <= i <= len(lattice.actions):
                sequence[i, 1 + lattice.actions[i - 1]] = 1.0
            else:
                sequence[i, 0] = 1.0
            if residue == "P":
                sequence[i, 5] = 1.0
            else:
                sequence[i, 6] = 1.0
        return {"sequence": sequence, "valid_moves": valid_move_mask(self.env)}


# Returns a function that gives a new random H/P sequence on every call
def random_sequence_fn(length, seed):
    
    # Its own seeded generator so the same seed always gives the same sequences
    rng = random.Random(seed)

    def next_sequence():
        return "".join(rng.choices("HP", k=length))

    return next_sequence


# Build the env used for training and evaluation
def make_env(seq_fn, sequence_view=False, collision_penalty=-2, trap_penalty=0.5, distance_penalty=0.01, max_steps_factor=4):
    
    env = Lattice2DEnv(seq_fn, collision_penalty=collision_penalty, trap_penalty=trap_penalty,  distance_penalty=distance_penalty)
    
    # Every sequence has the same length so the first one sets the step limit
    seq_length = len(env.seq)

    env = GuardAgainstLatticeTrap(env)
    env = TimeLimit(env, max_episode_steps=max_steps_factor * seq_length)
    env = FoldMetrics(env)
    
    # Swap the grid for the one row per residue view (used by the LSTM)
    if sequence_view:
        env = SequenceObservation(env)

    return env
