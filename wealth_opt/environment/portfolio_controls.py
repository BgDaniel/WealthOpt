import numpy as np
from typing import List, Tuple

from wealth_opt.environment.interfaces.controls import Controls
from wealth_opt.environment.interfaces.states import Coords


class PortfolioControls(Controls):
    """
    Concrete class returning admissible portfolio control coordinate meshes
    (Δu_1, ..., Δu_n, c) for a given state.
    """

    def __init__(self, dim: int, delta_u_max: float = 0.1,
                 c_frac_max: float = 0.2, n_steps: int = 10, allow_short: bool = False):
        """
        Initialize portfolio control parameters.

        Parameters
        ----------
        dim : int
            Number of assets.
        states : States
            Reference state space.
        delta_u_max : float
            Maximum absolute change per asset weight.
        c_frac_max : float
            Maximum consumption as fraction of current wealth.
        n_steps : int
            Number of discretization steps per control dimension.
        allow_short : bool
            Whether short-selling is allowed.
        """
        self.delta_u_max = delta_u_max
        self.c_frac_max = c_frac_max
        self.n_steps = n_steps
        self.allow_short = allow_short

    def admissible_controls_mesh(
        self,
        current_u: np.ndarray,
        current_S: np.ndarray
    ) -> Coords:
        """
        Generate coordinate meshes of admissible controls (Δu_1,...,Δu_n,c)
        for the given portfolio state.

        Parameters
        ----------
        current_u : np.ndarray
            Current portfolio weights (sum = 1).
        current_S : np.ndarray
            Current asset values.

        Returns
        -------
        Coords
            List of ndarrays (length n_assets+1), each of shape
            (n_steps, ..., n_steps), representing the coordinate mesh for
            each control variable (Δu_1, ..., Δu_n, c). Invalid combinations
            are replaced with np.nan.
        """
        # Total wealth
        wealth = np.sum(current_u * current_S)

        # Discretized ranges for Δu_i
        delta_ranges = [np.linspace(-self.delta_u_max, self.delta_u_max, self.n_steps)
                        for _ in range(self.dim)]
        # Discretized range for consumption
        c_range = np.linspace(0.0, self.c_frac_max * wealth, self.n_steps)

        # Full n+1-dimensional coordinate mesh
        mesh: Coords = list(np.meshgrid(*delta_ranges, c_range, indexing='ij'))

        # Mask invalid combinations
        for idx in np.ndindex(mesh[0].shape):
            delta_u = np.array([m[idx] for m in mesh[:-1]])
            c = mesh[-1][idx]
            new_u = current_u + delta_u

            # No short-selling
            if not self.allow_short and np.any(new_u < 0):
                for m in mesh:
                    m[idx] = np.nan
                continue

            # Normalize weights
            new_u = new_u / np.sum(new_u)

            # Validate sum=1 and positivity
            if np.any(new_u < 0) or not np.isclose(np.sum(new_u), 1.0, atol=1e-8):
                for m in mesh:
                    m[idx] = np.nan
                continue

            # Validate that new wealth after consumption is non-negative
            new_wealth = np.sum(new_u * current_S) - c
            if c < 0 or new_wealth < 0:
                for m in mesh:
                    m[idx] = np.nan

        return mesh

    def trivial_control(self) -> Coords:
        """
        Return a static control as a coordinate mesh (Δu_i = 0, c = 0)
        compatible with the Coords format (list of ndarrays).

        Returns
        -------
        Coords
            List of ndarrays of shape (1, 1, ..., 1) with n_assets+1 elements,
            each filled with 0.0 (Δu_i or c).
        """
        shape = (1,) * (self.dim + 1)  # create singleton mesh
        return [np.zeros(shape) for _ in range(self.dim + 1)]
