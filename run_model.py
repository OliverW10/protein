from stable_baselines3 import PPO
import random
from gym_lattice.envs import Lattice2DEnv
import numpy as np
from stable_baselines3 import PPO


np.random.seed(42)

def random_seq_fn(length = 7):
    return ''.join(random.choices('HP', k=length))
env = Lattice2DEnv(random_seq_fn, end_ceneterd=True, distance_penalty=0.01)

model = PPO.load("last.model")

def run_with_seq(seq: str):
    env.seq_fn = lambda: seq
    obs, _ = env.reset()
    print("Solving for:", env.seq)
    env.render()
    while not env.done:
        # Random agent samples from action space
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        env.render()
        if terminated or truncated:
            print("Episode finished! Reward: {} | Collisions: {} | Actions: {}".format(reward, info['collisions'], info['actions']))
            break

run_with_seq("HPPH")

for i_episodes in range(3):
    run_with_seq(random_seq_fn(10))