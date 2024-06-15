import numpy as np
from dataclasses import dataclass
from typing import Optional, Union


@dataclass()
class GeneralModelConfig:
    color: str


@dataclass(init=False, repr=False)
class EpsilonGreedyModelConfig(GeneralModelConfig):
    epsilon: float
    alpha: Union[float, str]
    q_start: Optional[np.ndarray] = None

    def __init__(self, epsilon, alpha, color, q_start=None):
        super().__init__(color)
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_start = q_start

    def __repr__(self):
        return f"Epsilon-Greedy -> epsilon: {self.epsilon}, alpha: {self.alpha}, color: {self.color}, q_start: {self.q_start}"


@dataclass(init=False, repr=False)
class UCBModelConfig(GeneralModelConfig):
    c: float
    alpha: Union[float, str]

    def __init__(self, c, alpha, color):
        super().__init__(color)
        self.c = c
        self.alpha = alpha

    def __repr__(self):
        return f"Upper Confidence Bound -> c: {self.c}, alpha: {self.alpha}, color: {self.color}"


@dataclass(init=False, repr=False)
class ThompsonSamplingModelConfig(GeneralModelConfig):
    prior_mean: float = 0.0
    prior_std: float = 100.0

    def __init__(self, prior_mean, prior_std, color):
        super().__init__(color)
        self.prior_mean = prior_mean
        self.prior_std = prior_std

    def __repr__(self):
        return f"Thompson Sampling -> mean: {self.prior_mean}, std: {self.prior_std}, color: {self.color}"


k = 10
runs = 2000
time_steps = 1000
