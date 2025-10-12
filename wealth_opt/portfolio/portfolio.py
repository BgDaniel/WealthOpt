import numpy as np
from abc import ABC
from typing import List

from wealth_opt.portfolio.assets.asset import Asset


class Portfolio(ABC):
    """
    Abstract base class representing a portfolio (control) of tradable assets.

    A portfolio combines multiple assets with given allocation weights or portfolio_control.
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
        """
        if not assets:
            raise ValueError("Portfolio must contain at least one asset.")
        self.assets = assets

    # --- Utility Property ---
    @property
    def n_assets(self) -> int:
        """Return the number of assets in the portfolio."""
        return len(self.assets)

    # --- Drift (μ_P) ---
    def mu(self, t: float, x: float, u: np.ndarray, c: float) -> float:
        """
        Compute the portfolio drift μ_P(t, x, u, c).

        The drift represents the expected rate of change of wealth:
            μ_P(t, x, u, c) = x * Σ_i [u_i * μ_i(t, x)] - c

        Parameters
        ----------
        t : float
            Current time.
        x : float
            Current portfolio wealth (state variable).
        u : np.ndarray
            Control (allocation) vector for the assets.
        c : float
            Instantaneous consumption rate.

        Returns
        -------
        float
            Portfolio drift value.
        """
        if len(u) != self.n_assets:
            raise ValueError("Control vector 'u' must match the number of assets in the portfolio.")

        mu_vals = np.array([asset.mu(t, x) for asset in self.assets])
        return x * np.dot(u, mu_vals) - c

    # --- Volatility (σ_P) ---
    def sigma(self, t: float, x: float, u: np.ndarray) -> float:
        """
        Compute the portfolio volatility σ_P(t, x, u).

        Assuming asset returns are independent, the portfolio volatility is:
            σ_P(t, x, u) = x * sqrt(Σ_i [u_i² * σ_i(t, x)²])

        Parameters
        ----------
        t : float
            Current time.
        x : float
            Current portfolio wealth.
        u : np.ndarray
            Control (allocation) vector for the assets.

        Returns
        -------
        float
            Portfolio volatility value.
        """
        if len(u) != self.n_assets:
            raise ValueError("Control vector 'u' must have the same length as the number of assets.")

        sigmas = np.array([asset.sigma(t, x) for asset in self.assets])
        return x * np.sqrt(np.dot(u**2, sigmas**2))
