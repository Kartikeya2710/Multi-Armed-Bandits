# Incremental implementation of sample averaging method for action value estimation where actions are selected using an epsilon-greedy strategy
from bandits_epsilon_greedy_agent import BanditsEpsilonGreedyAgent
import constants as c

agent = BanditsEpsilonGreedyAgent(c.k, c.epsilons, c.runs, c.time_steps, c.colors)

agent.train()

agent.visualize()
