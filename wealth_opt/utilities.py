import numpy as np

def crra(c, gamma=2.0):
    """Constant Relative Risk Aversion utility."""
    return (c ** (1 - gamma)) / (1 - gamma) if c > 0 else -np.inf

def log_utility(c):
    """Log utility function."""
    return np.log(c) if c > 0 else -np.inf

def square_root(c):
    """Square root utility."""
    return np.sqrt(c) if c > 0 else -np.inf