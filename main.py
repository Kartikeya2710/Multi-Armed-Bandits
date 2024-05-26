# Incremental implementation of sample averaging method for action value estimation where actions are selected using an epsilon-greedy strategy (SAEG)
from agent import BanditAgent
import config

model_configs = [
    config.EpsilonGreedyModelConfig(0.1, "avg", "red"),
    config.UCBModelConfig(2, "avg" "blue"),
]

agent = BanditAgent(config.k, model_configs, config.runs, config.time_steps)

agent.train()

agent.visualize()
