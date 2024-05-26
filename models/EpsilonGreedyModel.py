import numpy as np
from bandits_env import BanditsEnv
from models.BaseModel import BaseModel
from utils.argmax_random import argmax_random


class EpsilonGreedyModel(BaseModel):
    def __init__(
        self,
        k: int,
        epsilon: float,
        step_size: float | str = "avg",
        time_steps: int = 1000,
    ):
        super().__init__(k, time_steps)
        self.epsilon = epsilon
        self.step_size = step_size
        self.alpha = step_size if isinstance(step_size, float) else None

        self.avg_rewards = np.zeros(shape=(self.time_steps), dtype=np.float64)
        self.optimal_action_selection = np.zeros(
            shape=self.time_steps, dtype=np.float64
        )

    def train_one_run(self, run_idx: int, bandits_env: BanditsEnv):
        self.q_t = np.zeros(shape=self.k, dtype=np.float64)

        self.action_freq = None
        if not self._constant_step_size():
            self.action_freq = np.zeros(shape=(self.k,), dtype=np.int32)

        optimal_action = np.argmax(bandits_env.true_values)

        for step in range(self.time_steps):

            action_idx = self._choose_action()

            r_t = bandits_env.sample_reward(action_idx)

            # update average reward and optimal action metrics
            self.avg_rewards[step] = (self.avg_rewards[step] * run_idx + r_t) / (
                run_idx + 1
            )
            self.optimal_action_selection[step] = (
                self.optimal_action_selection[step] * run_idx
                + (action_idx == optimal_action) * 100
            ) / (run_idx + 1)

            if not self._constant_step_size():
                self.action_freq[action_idx] += 1
                self.alpha = 1.0 / self.action_freq[action_idx]

            self.q_t[action_idx] = self.q_t[action_idx] + self.alpha * (
                r_t - self.q_t[action_idx]
            )

    def _choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(low=0, high=self.k)

        return argmax_random(self.q_t)

    def _constant_step_size(self):
        return isinstance(self.step_size, int)

    def __str__(self):
        return f"eg: eps={self.epsilon} alpha={self.step_size}"
