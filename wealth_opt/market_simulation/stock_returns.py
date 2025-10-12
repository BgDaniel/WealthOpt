import numpy as np
import pandas as pd
from typing import Tuple
from wealth_opt.portfolio.assets.stock import Stock


class StockReturns(Stock):
    """
    Geometric Brownian Motion (GBM) stock return simulator.

    Inherits from `Stock`, and extends it with functionality to simulate
    daily cumulative returns across multiple Monte Carlo paths.

    Dynamics:
        dS_t = μ * S_t * dt + σ * S_t * dW_t
    """

    def __init__(
        self,
        as_of_date: pd.Timestamp,
        simulation_days: pd.DatetimeIndex,
        r: float = 0.0,
        sigma: float = 0.1,
    ) -> None:
        """
        Initialize the StockReturns GBM model.

        Parameters
        ----------
        as_of_date : pd.Timestamp
            Start date of the simulation.
        simulation_days : pd.DatetimeIndex
            Array of simulation dates.
        r : float
            Expected return (drift).
        sigma : float
            Volatility of the stock.
        """
        if simulation_days[0] != as_of_date:
            raise ValueError("as_of_date must equal the first simulation day")

        super().__init__(mu=r, sigma=sigma)  # use Stock constructor

        self.as_of_date = as_of_date
        self.simulation_days = simulation_days
        self.dt = 1 / 252  # assume 252 trading days per year

    def simulate(self, n_sims: int) -> pd.DataFrame:
        """
        Simulate cumulative returns using GBM dynamics.

        Parameters
        ----------
        n_sims : int
            Number of Monte Carlo simulation paths.

        Returns
        -------
        pd.DataFrame
            Simulated cumulative returns with shape (len(simulation_days), n_sims),
            starting at 1.0.
        """
        n_steps = len(self.simulation_days)
        sims = np.zeros((n_steps, n_sims))
        sims[0, :] = 1.0  # start normalized at 1.0

        for t in range(1, n_steps):
            Z = np.random.normal(size=n_sims)
            sims[t, :] = sims[t - 1, :] * np.exp(
                (self.mu_const - 0.5 * self.sigma_const**2) * self.dt
                + self.sigma_const * np.sqrt(self.dt) * Z
            )

        return pd.DataFrame(
            sims,
            index=self.simulation_days,
            columns=[f"sim_{i}" for i in range(n_sims)],
        )
