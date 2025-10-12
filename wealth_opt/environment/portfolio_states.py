import numpy as np
from typing import List, Optional

from wealth_opt.environment.interfaces.states import States, Coords


class PortfolioStates(States):
    """
    Concrete implementation of StateBase using a uniform mesh grid.

    Each asset dimension is discretized from 0 to its maximum value,
    and the full state space is represented as a flattened mesh.
    """

    def __init__(
        self,
        n_assets: int,
        max_asset: Optional[List[float]] = None,
        n_steps: int = 500
    ) -> None:
        """
        Initialize the uniform state space for a portfolio.

        Parameters
        ----------
        n_assets : int
            Number of assets in the portfolio.
        max_asset : Optional[List[float]], optional
            Maximum values for each asset dimension. Defaults to 1000.0 for each asset.
        n_steps : int, optional
            Number of grid steps per asset dimension. Default is 500.
        """
        self.n_assets: int = n_assets
        self.n_steps: int = n_steps
        self._max_asset: List[float] = max_asset if max_asset is not None else [1000.0] * n_assets

        if len(self._max_asset) != n_assets:
            raise ValueError("Length of max_asset must equal n_assets.")

        self._x: np.ndarray = self._build_mesh()

    def _build_mesh(self) -> np.ndarray:
        """
        Build a uniform n-dimensional mesh grid of portfolio asset states.

        Returns
        -------
        np.ndarray
            Flattened array of shape (n_steps^n_assets, n_assets),
            representing all combinations of asset states.
        """
        coords = [np.linspace(0.0, a_max, self.n_steps) for a_max in self._max_asset]
        mesh = np.meshgrid(*coords, indexing="ij")
        return np.stack([m.flatten() for m in mesh], axis=-1)

    @property
    def x(self) -> Coords:
        """
        Access the flattened state mesh.

        Returns
        -------
        Coords
            List containing a single ndarray representing the flattened
            state space of shape (n_steps^n_assets, n_assets).
        """
        return [self._x]
