from abc import ABC, abstractmethod
from typing import List
import numpy as np

# Type alias for a list of coordinate arrays
Coords = List[np.ndarray]


class States(ABC):
    """
    Abstract base class representing a discretized state space.

    Each asset dimension is defined on [0, a_max_asset[i]], and
    the full state space can be represented as coordinate meshes.
    """

    @property
    @abstractmethod
    def x(self) -> Coords:
        """
        Access the state mesh coordinates.

        Returns
        -------
        Coords
            List of ndarrays representing coordinate meshes for each dimension.
            Each array has shape (n_steps, ..., n_steps), depending on the number of dimensions.
        """
        raise NotImplementedError()
