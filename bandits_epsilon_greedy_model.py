import numpy as np
from bandits_env import BanditsEnv


class BanditsEpsilonGreedyModel:
    def __init__(self, k, epsilon, time_steps=1000):
        self.k = k
        self.epsilon = epsilon
        self.time_steps = time_steps
        self.avg_rewards = np.zeros(shape=(self.time_steps), dtype=np.float64)
        self.optimal_action_selection = np.zeros(
            shape=(self.time_steps), dtype=np.float64
        )

    def train_one_run(self, run_idx, bandits_env: BanditsEnv):
        q_t = np.zeros(shape=(self.k,), dtype=np.float64)
        action_freq = np.zeros(shape=(self.k,), dtype=np.int32)
        optimal_action = np.argmax(bandits_env.true_values)

        for step in range(0, self.time_steps):
            action_idx = -1

            # select action
            if np.random.uniform(low=0.0, high=1.0) < self.epsilon:
                action_idx = np.random.randint(low=0, high=self.k)
            else:
                action_idx = np.argmax(q_t)

            action_freq[action_idx] += 1

            # sample reward from the arm
            r_t = bandits_env.sample_reward(action_idx)

            # update average reward and optimal action metrics
            self.avg_rewards[step] = (self.avg_rewards[step] * run_idx + r_t) / (
                run_idx + 1
            )
            self.optimal_action_selection[step] = (
                self.optimal_action_selection[step] * run_idx
                + (action_idx == optimal_action)
            ) / (run_idx + 1)

            q_t[action_idx] = (
                q_t[action_idx] + (r_t - q_t[action_idx]) / action_freq[action_idx]
            )
