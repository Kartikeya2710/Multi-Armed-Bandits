import numpy as np
from bandits_env import BanditsEnv
from utils.argmax_random import argmax_random
from models.BaseModel import BaseModel


# Assuming the action rewards are sampled from a gaussian distribution, our priors are also gonna be gaussian distributions
class ThompsonSamplingModel(BaseModel):
    def __init__(self, prior_mean: float, prior_std: float, k: int, time_steps: int):
        super().__init__(k, time_steps)

        if not self._valid_prior_std(prior_std):
            raise Exception(f"prior_std must be positive, received {prior_std}")

        self.prior_mean = prior_mean
        self.prior_std = prior_std

        self.generators = [np.random.default_rng() for _ in range(k)]

    def _reward_estimate(self) -> None:
        self.mean_rew_est = np.empty(self.k)
        for i in range(self.k):
            self.mean_rew_est[i] = self.generators[i].normal(
                self.means[i], self.stds[i]
            )

    def _choose_action(
        self,
    ) -> int:
        self._reward_estimate()
        return argmax_random(self.mean_rew_est)

    def train_one_run(self, run_idx: int, bandits_env: BanditsEnv) -> None:

        optimal_action = np.argmax(bandits_env.true_values)
        # print(f"optimal action: {optimal_action}")
        self.action_freq = np.zeros(shape=(self.k,), dtype=np.int32)
        self.accum_reward = np.zeros(shape=(self.k), dtype=np.float64)
        self.means = np.full(self.k, self.prior_mean, dtype=np.float64)
        self.stds = np.full(self.k, self.prior_std, dtype=np.float64)

        # print(f"True means: {bandits_env.true_values}")

        for step in range(self.time_steps):
            action_idx = self._choose_action()
            self.action_freq[action_idx] += 1

            r_t = bandits_env.sample_reward(action_idx)

            self.avg_rewards[step] = (self.avg_rewards[step] * run_idx + r_t) / (
                run_idx + 1
            )
            self.optimal_action_selection[step] = (
                self.optimal_action_selection[step] * run_idx
                + (action_idx == optimal_action) * 100
            ) / (run_idx + 1)
            self.accum_reward[action_idx] += r_t

            # for the action performed, you estimated self.mean_rew_est[action_idx] but actually got r_t, so now you correct your estimate!

            tau_est = 1.0 / np.square(self.stds[action_idx])

            # print(
            #     f"{action_idx} -> mean {self.means[action_idx]}",
            #     end=" ",
            # )

            # mean_est = (tau_est*mean_est + tau_true * accum_rewards[action]) / (tau_est + n*tau_true)
            self.means[action_idx] = (
                (tau_est * self.means[action_idx]) + (1 * self.accum_reward[action_idx])
            ) / (tau_est + self.action_freq[action_idx] * 1)

            # print(
            #     f"to {self.means[action_idx]}, std {self.stds[action_idx]}",
            #     end=" ",
            # )
            # tau_est = tau_est + n * tau_true, where tau_true = 1/std_true**2
            tau_est += self.action_freq[action_idx] * 1
            self.stds[action_idx] = 1.0 / np.sqrt(tau_est)

            # print(f"to {self.stds[action_idx]}")


    def _valid_prior_std(self, prior_std: float) -> bool:
        return prior_std > 0

    def __str__(self):
        return f"thompson: mean={self.prior_mean} std={self.prior_std}"
