import matplotlib.pyplot as plt
from models.UCBModel import UCBModel
from models.EpsilonGreedyModel import EpsilonGreedyModel
from models.ThompsonSamplingModel import ThompsonSamplingModel
from bandits_env import BanditsEnv
from config import (
    GeneralModelConfig,
    EpsilonGreedyModelConfig,
    UCBModelConfig,
    ThompsonSamplingModelConfig,
)
from tqdm import tqdm


class BanditAgent:
    def __init__(
        self, k: int, configs: list[GeneralModelConfig], runs: int, time_steps: int
    ):
        self.num_models = len(configs)
        self.models = []
        self.colors = []

        for config in configs:

            print(config)
            if isinstance(config, EpsilonGreedyModelConfig):
                self.models.append(
                    EpsilonGreedyModel(
                        k,
                        config.epsilon,
                        config.alpha,
                        config.q_start,
                        time_steps,
                    )
                )
            elif isinstance(config, UCBModelConfig):
                self.models.append(UCBModel(k, config.c, config.alpha, time_steps))

            elif isinstance(config, ThompsonSamplingModelConfig):
                self.models.append(
                    ThompsonSamplingModel(
                        config.prior_mean, config.prior_std, k, time_steps
                    )
                )

            self.colors.append(config.color)

        self.k = k
        self.runs = runs

    def train(self):
        for run_idx in tqdm(
            range(self.runs), desc=f"Training {self.num_models} models"
        ):
            bandits_env = BanditsEnv(self.k)

            for model in self.models:
                model.train_one_run(run_idx, bandits_env)

    def visualize(self):
        """To be called after training for visualizing average rewards obtained by the model"""
        legend = [model.__str__() for model in self.models]
        for idx, model in enumerate(self.models):
            plt.plot(model.avg_rewards, self.colors[idx])

        plt.title(f"Average reward v/s Steps - {self.runs} runs")
        plt.ylabel("Average reward")
        plt.xlabel("Steps")
        plt.ylim(0, 1.8)
        plt.legend(legend, loc="lower right")
        plt.show()

        for idx, model in enumerate(self.models):
            plt.plot(model.optimal_action_selection, self.colors[idx])

        plt.title(f"% Optimal action v/s Steps - {self.runs} runs")
        plt.ylabel("% Optimal action")
        plt.xlabel("Steps")
        plt.ylim(0, 100)
        plt.legend(legend, loc="lower right")
        plt.show()
