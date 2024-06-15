import numpy as np
from bandits_env import BanditsEnv
from models.BaseModel import BaseModel
from utils.argmax_random import argmax_random


class UCBModel(BaseModel):
    def __init__(
        self,
        k: int,
        c: float,
        step_size: float | str = "avg",
        time_steps: int = 1000,
    ):
        super().__init__(k, time_steps)
        self.c = c
        self.step_size = step_size
        self.alpha = step_size if isinstance(step_size, float) else None

    def train_one_run(self, run_idx: int, bandits_env: BanditsEnv):
        self.q_t = np.zeros(shape=self.k, dtype=np.float64)
        # initialized as ones to prevent dividing by zero during action selection
        self.action_freq = np.ones(shape=self.k, dtype=np.float64)

        optimal_action = np.argmax(bandits_env.true_values)

        for step in range(self.time_steps):

            action_idx = self._choose_action(step)
            r_t = bandits_env.sample_reward(action_idx)

            self.avg_rewards[step] = (self.avg_rewards[step] * run_idx + r_t) / (
                run_idx + 1
            )

            self.optimal_action_selection[step] = (
                self.optimal_action_selection[step] * run_idx
                + (action_idx == optimal_action) * 100
            ) / (run_idx + 1)

            if not self._constant_step_size():
                self.alpha = 1.0 / (self.action_freq[action_idx])

            self.q_t[action_idx] = self.q_t[action_idx] + self.alpha * (
                r_t - self.q_t[action_idx]
            )

            self.action_freq[action_idx] += 1

    # this method signature allows proper overriding of _choose_action or else the method signature will differ and will lead to issues when trying to use UCBModel polymorphically as BaseModel
    def _choose_action(self, step, *args, **kwargs) -> int:
        # choose randomly if you have just started
        if step == 0:
            return argmax_random(self.q_t)

        # A_t = argmax{a} ( q_t(a) + c * sqrt(ln(t) / N_t(a)) )

        return argmax_random(
            self.q_t + self.c * np.sqrt(np.log(step + 1) / self.action_freq)
        )

    def _constant_step_size(self):
        return isinstance(self.step_size, int)

    def __str__(self):
        return f"ucb: c={self.c} α={self.step_size}"
