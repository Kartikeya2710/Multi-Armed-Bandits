import numpy as np
from bandits_env import BanditsEnv


class BaseModel:
    def __init__(self, k, time_steps):
        self.k = k
        self.time_steps = time_steps
        self.avg_rewards = np.zeros(shape=self.time_steps, dtype=np.float64)
        self.optimal_action_selection = np.zeros(
            shape=self.time_steps, dtype=np.float64
        )

    def train_one_run(self, run_idx: int, bandits_env: BanditsEnv) -> None:
        raise NotImplementedError(
            "user must define train_one_run() to use BaseModel class"
        )

    def _choose_action(self, *args, **kwargs) -> int:
        raise NotImplementedError(
            "user must define _choose_action() to use BaseModel class"
        )
