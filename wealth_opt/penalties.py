import numpy as np


def quadratic_penalty(x: np.ndarray, target: float = 0.0) -> np.ndarray:
    return (x - target) ** 2


def linear_penalty(x: np.ndarray, target: float = 0.0) -> np.ndarray:
    return np.abs(x - target)
