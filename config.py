from dataclasses import dataclass
from typing import Union


@dataclass()
class GeneralModelConfig:
    color: str


@dataclass(init=False, repr=False)
class EpsilonGreedyModelConfig(GeneralModelConfig):
    epsilon: float
    alpha: Union[float, str]

    def __init__(self, epsilon, alpha, color):
        super().__init__(color)
        self.epsilon = epsilon
        self.alpha = alpha

    def __repr__(self):
        return f"Epsilon-Greedy -> epsilon: {self.epsilon}, alpha: {self.alpha}, color: {self.color}"


@dataclass(init=False, repr=False)
class UCBModelConfig(GeneralModelConfig):
    uncertainity: float
    alpha: Union[float, str]

    def __init__(self, uncertainity, alpha, color):
        super().__init__(color)
        self.uncertainity = uncertainity
        self.alpha = alpha

    def __repr__(self):
        return f"Upper Confidence Bound -> uncertainity: {self.uncertainity}, alpha: {self.alpha}, color: {self.color}"


k = 10
runs = 2000
time_steps = 1000
