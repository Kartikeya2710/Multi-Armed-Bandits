# Multi-Armed-Bandits
This repository contains minimalistic implementations of simple reinforcement learning algorithms for solving multi-armed bandits

*Note: All implementations use 10 arms*

## To do
1. Implement decaying epsilon for epsilon-greedy model
2. Figure out a way to store, calculate and display metrics

### Training configuration

1. Number of runs = *2000*
2. Number of time steps in each run = *1000*

## Epsilon-Greedy Model

Incremental implementation of sample averaging and constant step size method with epsilon-greedy action selection strategy from *Richard S. Sutton and Andrew G. Barto - Reinforcement Learning (2nd edition)*

<img src="./plots/epsilon_greedy/avg_reward.png" alt="average reward v/s steps" width=400>

<img src="./plots/epsilon_greedy/optimal_action.png" alt="% optimal action v/s steps" width=400>


## Upper Confidence Bound Model

Implementation of upper confidence bound action selection strategy and constant step size from *Richard S. Sutton and Andrew G. Barto - Reinforcement Learning (2nd edition)*

<img src="./plots/ucb/avg_reward.png" alt="average reward v/s steps" width=400>

<img src="./plots/ucb/optimal_action.png" alt="% optimal action v/s steps" width=400>

### Comparisons

<img src="./plots/comparisons/eg_vs_ucb_reward.png" alt="average reward v/s steps" width=400>

<img src="./plots/comparisons/eg_vs_ucb_opt.png" alt="% optimal action v/s steps" width=400>