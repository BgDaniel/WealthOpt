import numpy as np
from abc import ABC
from typing import List

from wealth_opt.portfolio.assets.asset import Asset

Weights = List[np.ndarray]


class Portfolio(ABC):
    """
    Abstract base class representing a portfolio (control) of tradable assets.

    A portfolio combines multiple assets with given allocation weights.
    The portfolio’s drift and volatility are derived from the dynamics of its constituent assets.

    Attributes
    ----------
    assets : List[Asset]
        List of asset instances included in the portfolio.
    """

    def __init__(self, assets: List[Asset]) -> None:
        """
        Initialize a portfolio of assets.

        Parameters
        ----------
        assets : List[Asset]
            List of asset instances composing the portfolio.

        Raises
        ------
        ValueError
            If the list of assets is empty.
        """
        if not assets:
            raise ValueError("Portfolio must contain at least one asset.")
        self.assets: List[Asset] = assets

    # -------------------------------------------------------------------------
    @property
    def n_assets(self) -> int:
        """
        Return the number of assets in the portfolio.

        Returns
        -------
        int
            Number of assets.
        """
        return len(self.assets)

    # -------------------------------------------------------------------------
    def mu(self, t: float, x: np.ndarray, u: Weights, c: np.ndarray) -> np.ndarray:
        """
        Compute the portfolio drift at given time and state for control u.

        Parameters
        ----------
        t : float
            Current time.
        x : np.ndarray
            1D array of wealth or state values.
        u : Weights
            List of arrays representing the control weights for each asset.
            Each element must have the same shape as `x`.
        c : np.ndarray
            Consumption array to subtract from drift, same shape as `x`.

        Returns
        -------
        np.ndarray
            Portfolio drift at each point in `x`.
        """
        mu_vals = np.zeros_like(x)
        for i_asset in range(len(self.assets)):
            mu_vals += self.assets[i_asset].mu(t) * x * u[i_asset]
        mu_vals -= c
        return mu_vals

    # -------------------------------------------------------------------------
    def sigma(self, t: float, x: np.ndarray, u: Weights) -> np.ndarray:
        """
        Compute the portfolio volatility at given time and state for control u.

        Parameters
        ----------
        t : float
            Current time.
        x : np.ndarray
            1D array of wealth or state values.
        u : Weights
            List of arrays representing the control weights for each asset.
            Each element must have the same shape as `x`.

        Returns
        -------
        np.ndarray
            Portfolio volatility at each point in `x`.
        """
        sigma_vals = np.zeros_like(x)
        for i_asset in range(len(self.assets)):
            sigma_vals += self.assets[i_asset].sigma(t) * x * u[i_asset]
        return sigma_vals
