import numpy as np
import pandas as pd
from typing import Tuple

class GBMReturns:
    """
    Geometric Brownian Motion (GBM) return simulator.

    Simulates daily cumulative returns using GBM with drift and volatility.
    """

    def __init__(
        self,
        as_of_date: pd.Timestamp,
        simulation_days: pd.DatetimeIndex,
        r: float = 0.0,
        sigma: float = 0.1,
    ) -> None:
        """
        Initialize the GBMReturns model.

        Parameters
        ----------
        as_of_date : pd.Timestamp
            Start date of the simulation.
        simulation_days : pd.DatetimeIndex
            Array of simulation dates.
        r : float
            Drift of the GBM.
        sigma : float
            Volatility of the GBM.
        """
        if simulation_days[0] != as_of_date:
            raise ValueError("as_of_date must equal the first simulation day")

        self.as_of_date = as_of_date
        self.simulation_days = simulation_days
        self.mu = r
        self.sigma = sigma
        self.dt = 1 / 252  # daily steps assuming 252 trading days per year

    def simulate(self, n_sims: int) -> pd.DataFrame:
        """
        Simulate cumulative returns using GBM.

        Parameters
        ----------
        n_sims : int
            Number of simulation paths.

        Returns
        -------
        pd.DataFrame
            Simulated cumulative returns with shape (len(simulation_days), n_sims),
            starting at 1.0.
        """
        n_steps = len(self.simulation_days)
        sims = np.zeros((n_steps, n_sims))
        sims[0, :] = 1.0  # start at 1.0

        # Generate random shocks
        for t in range(1, n_steps):
            Z = np.random.normal(size=n_sims)
            sims[t, :] = sims[t - 1, :] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z
            )

        return pd.DataFrame(sims, index=self.simulation_days, columns=[f"sim_{i}" for i in range(n_sims)])
