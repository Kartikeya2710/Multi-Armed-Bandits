import numpy as np
import matplotlib.pyplot as plt
from bandits_epsilon_greedy_model import BanditsEpsilonGreedyModel
from bandits_env import BanditsEnv
from tqdm import tqdm


class BanditsEpsilonGreedyAgent:
    def __init__(
        self, k: int, epsilons: list, runs: int, time_steps: int, colors: list
    ):
        self.num_models = len(epsilons)
        self.models = [
            BanditsEpsilonGreedyModel(k, epsilons[i], time_steps)
            for i in range(self.num_models)
        ]
        self.k = k
        self.epsilons = epsilons
        self.runs = runs
        self.colors = colors

    def train(self):
        for run_idx in tqdm(
            range(self.runs), desc=f"Training {self.num_models} models"
        ):
            bandits_env = BanditsEnv(self.k)

            for model in self.models:
                model.train_one_run(run_idx, bandits_env)

    def visualize(self):
        """To be called after training for visualizing average rewards obtained by the model"""
        for idx, model in enumerate(self.models):
            plt.plot(model.avg_rewards, self.colors[idx])

        plt.title("Average reward v/s Steps")
        plt.ylabel("Average reward")
        plt.xlabel("Steps")
        plt.legend(self.epsilons, title="epsilon")
        plt.show()

        for idx, model in enumerate(self.models):
            plt.plot(model.optimal_action_selection, self.colors[idx])

        plt.title("% Optimal action v/s Steps")
        plt.ylabel("% Optimal action")
        plt.xlabel("Steps")
        plt.legend(self.epsilons, title="epsilon")
        plt.show()
