import numpy as np


def argmax_random(array: np.ndarray) -> int:
    max_value = np.max(array)
    max_indices = np.where(array == max_value)[0]
    return np.random.choice(max_indices)
