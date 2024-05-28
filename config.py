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


k = 10
runs = 2000
time_steps = 1000
