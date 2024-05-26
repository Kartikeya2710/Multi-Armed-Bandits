from bandits_env import BanditsEnv


class BaseModel:
    def __init__(self, k, time_steps):
        self.k = k
        self.time_steps = time_steps

    def train_one_run(self, run_idx: int, bandits_env: BanditsEnv) -> None:
        raise NotImplementedError(
            "user must define train_one_run() to use BaseModel class"
        )

    def _choose_action(self, *args, **kwargs) -> int:
        raise NotImplementedError(
            "user must define _choose_action() to use BaseModel class"
        )

    def _constant_step_size(self) -> bool:
        raise NotImplementedError(
            "user must define _constant_step_size() to use BaseModel class"
        )
