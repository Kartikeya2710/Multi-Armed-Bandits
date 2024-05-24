import numpy as np

class BanditsEnv:
    def __init__(self, k):
        self.k = k
        self._create_distributions()

    def _create_distributions(self) -> None:
        self.generator = np.random.default_rng()
        self.true_values = self.generator.normal(loc=0, scale=1, size=self.k)

    def sample_reward(self, index) -> float:
        return self.generator.normal(self.true_values[index])
