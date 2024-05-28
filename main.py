from agent import BanditAgent
import numpy as np
import config

model_configs = [
    config.EpsilonGreedyModelConfig(0.0, 0.1, "blue", np.array([5.0] * config.k)),
    config.EpsilonGreedyModelConfig(0.1, 0.1, "grey"),
    config.UCBModelConfig(1, 0.1, "black"),
]

agent = BanditAgent(config.k, model_configs, config.runs, config.time_steps)

agent.train()

agent.visualize()
