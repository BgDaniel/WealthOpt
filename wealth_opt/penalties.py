import numpy as np

def quadratic_penalty(wealth, target=1_000_000):
    """Quadratic penalty relative to target wealth."""
    return (wealth - target) ** 2

def linear_penalty(wealth, target=1_000_000):
    """Linear penalty relative to target wealth."""
    return np.abs(wealth - target)