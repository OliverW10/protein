import math
import csv
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _exponential_epsilon(progress, eps_min, eps_max, eps_decay):
    progress = min(max(progress, 0.0), 1.0)
    decay = math.exp(-eps_decay * progress)
    return eps_min + (eps_max - eps_min) * decay


# Epsilon that decays once per finished episode instead of every step
class EpisodeExponentialDecay:

    def __init__(self, total_episodes, eps_min, eps_max, eps_decay):

        self.total_episodes = total_episodes
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        # Counted up by the callback whenever an episode ends
        self.episodes = 0

    def __call__(self, progress_remaining):
        progress = self.episodes / self.total_episodes
        return _exponential_epsilon(
            progress, self.eps_min, self.eps_max, self.eps_decay
        )


class TimestepExponentialDecay:

    def __init__(self, eps_min, eps_max, eps_decay):
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay

    def __call__(self, progress_remaining):
        progress = 1.0 - progress_remaining
        return _exponential_epsilon(
            progress, self.eps_min, self.eps_max, self.eps_decay
        )


# Counts finished episodes so the schedule knows which episode training is on
class EpisodeEpsilonCallback(BaseCallback):

    def __init__(self, schedule):

        super().__init__()
        self.schedule = schedule

    def _init_callback(self):

        # Swap SB3's per-step schedule for the per-episode one when training starts
        self.model.exploration_schedule = self.schedule

    def _on_step(self):

        # True when the episode just ended
        if self.locals["dones"][0]:
            self.schedule.episodes += 1
        return True


# Fold one sequence with the current policy and no random moves
def greedy_episode(model, env):

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break

    # FoldMetrics put these in info when the episode ended
    return {"contacts": info["contacts"], "complete": info["complete"]}


# Every eval_every episodes fold the validation sequences greedily and keep the best model
class GreedyEvalCallback(BaseCallback):

    def __init__(self, validation_envs, run_dir, eval_every):

        super().__init__()
        self.validation_envs = validation_envs
        self.run_dir = run_dir
        self.eval_every = eval_every
        self.episodes = 0
        self.best_score = -1
        self.history = []

    def _on_step(self):

        if self.locals["dones"][0]:
            self.episodes += 1
            if self.episodes % self.eval_every == 0:
                self.evaluate()
        return True

    def evaluate(self):

        # Contacts only count when the fold is complete
        scores = []
        for env in self.validation_envs:
            result = greedy_episode(self.model, env)
            scores.append(result["contacts"] if result["complete"] else 0)

        score = np.mean(scores)
        self.history.append({"episodes": self.episodes, "score": score})

        # Saved every time so the learning curve survives if training stops early
        with open(
            os.path.join(self.run_dir, "eval_history.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as history_file:
            writer = csv.DictWriter(history_file, fieldnames=["episodes", "score"])
            writer.writeheader()
            writer.writerows(self.history)
        
        self.logger.record("eval/score", score)

        if score > self.best_score:
            self.best_score = score
            self.model.save(os.path.join(self.run_dir, "best_model"))

        print(f"episode {self.episodes}: score {score:.2f} contacts")


# From the Week 10 lab to save a checkpoint every save_interval steps
class SaveOnIntervalCallback(BaseCallback):
    def __init__(self, save_interval: int, save_path: str):
        super().__init__()
        self.save_interval = save_interval
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_interval == 0:
            save_file = os.path.join(self.save_path, f"model_{self.num_timesteps}")
            self.model.save(save_file)
        return True
