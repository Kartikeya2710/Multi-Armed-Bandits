from typing import Optional
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
        q_start: Optional[np.ndarray] = None,
        time_steps: int = 1000,
    ):
        super().__init__(k, time_steps)
        self.epsilon = epsilon
        self.alpha = step_size if isinstance(step_size, float) else None
        self.step_size = step_size
        self.q_start = q_start

        if not self._valid_q_start():
            raise Exception(
                f"found q_start of shape {q_start.shape} required shape is ({self.k},)"
            )

        if not self._valid_step_size():
            raise Exception(
                f"found step_size = {step_size} whereas step_size can be float or avg"
            )

    def train_one_run(self, run_idx: int, bandits_env: BanditsEnv):
        self.q_t = (
            self.q_start.copy()
            if self._q_start_provided()
            else np.zeros(shape=self.k, dtype=np.float64)
        )

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

            self.q_t[action_idx] += self.alpha * (r_t - self.q_t[action_idx])

    def _choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(low=0, high=self.k)

        return argmax_random(self.q_t)

    def _constant_step_size(self):
        return isinstance(self.step_size, (float, int))

    def _valid_step_size(self) -> bool:
        return isinstance(self.step_size, float) or self.step_size == "avg"

    def _valid_q_start(self) -> bool:
        return self.q_start is None or self.q_start.shape == (self.k,)

    def _q_start_provided(self) -> bool:
        return self.q_start is not None and self._valid_q_start()

    def __str__(self):
        return (
            f"eg: ε={self.epsilon} α={self.step_size} Q1=0"
            if self.q_start is None
            else f"eg: ε={self.epsilon} α={self.step_size} Q1={self.q_start[0]}"
        )
