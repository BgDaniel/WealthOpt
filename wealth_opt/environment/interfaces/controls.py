from abc import ABC, abstractmethod
from typing import List
import numpy as np
from wealth_opt.environment.interfaces.states import Coords


class Controls(ABC):
    """
    Abstract base class for portfolio controls.

    Subclasses must implement a method to generate admissible controls
    for a given portfolio state. Controls are represented as coordinate meshes,
    e.g., one mesh per control dimension (Δu_1,...,Δu_n,c).
    """

    @abstractmethod
    def admissible_controls(
        self,
        current_u: np.ndarray,
        current_S: np.ndarray
    ) -> Coords:
        """
        Return admissible controls for a given portfolio state as coordinate meshes.

        Parameters
        ----------
        current_u : np.ndarray
            Current portfolio weights (sum = 1).
        current_S : np.ndarray
            Current asset values.

        Returns
        -------
        Coords
            List of ndarrays (length n_assets+1), each representing a coordinate
            mesh of admissible values for that control dimension (Δu_i or c).
            Invalid combinations should be masked with np.nan.
        """
        raise NotImplementedError()

    @abstractmethod
    def trivial_control(self) -> Coords:
        """
        Return a fixed/default control vector that does not depend on the current state.

        This can be used as an initial guess or a default policy.

        Returns
        -------
        Coords
            Control vector of length n_assets+1: [Δu_1, ..., Δu_n, c].
        """
        raise NotImplementedError()
