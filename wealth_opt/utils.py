import numpy as np
from typing import Tuple, Callable


def broadcast_to_mesh(func: Callable) -> Callable:
    """
    Decorator to broadcast a function of `w` to the full coordinate mesh shape.

    The decorated function should accept `coords` (tuple of u, s, w) and other kwargs.
    """

    def wrapper(coords: Tuple[np.ndarray, np.ndarray, np.ndarray], *args, **kwargs) -> np.ndarray:
        u, s, w = coords
        # Compute function along w only
        result = func(w, *args, **kwargs)
        # Broadcast to full mesh shape
        full_shape = (u.shape[0], s.shape[1], w.shape[2])
        return np.broadcast_to(result, full_shape)

    return wrapper