import numpy as np
import pandas as pd
from typing import Callable
from wealth_opt.utilities import log_utility, crra
from wealth_opt.penalties import quadratic_penalty, linear_penalty
from wealth_opt.market_simulation.gbm_returns import GBMReturns


class PortfolioSimulation:
    """
    Holds all parameters, time grids, cashflows, and GBM simulations
    for a portfolio simulation environment. No optimization yet.
    """

    def __init__(
        self,
        as_of_date: pd.Timestamp,
        simulation_end_date: pd.Timestamp,
        retirement_date: pd.Timestamp,
        monthly_salary: float,
        monthly_pension: float,
        monthly_rent: float,
        r: float,
        sigma: float,
        instantaneous_utility: Callable[[float], float] = crra,
        terminal_penalty: Callable[[float], float] = quadratic_penalty,
    ):
        """
        Initialize the portfolio simulation environment.

        Parameters
        ----------
        as_of_date : pd.Timestamp
            Start date of the simulation.
        simulation_end_date : pd.Timestamp
            End date of the simulation.
        retirement_date : pd.Timestamp
            Retirement date when salary stops and pension starts.
        monthly_salary : float
            Monthly salary inflow before retirement.
        monthly_pension : float
            Monthly pension inflow after retirement.
        monthly_rent : float
            Monthly rent outflow.
        r : float
            Risk-free interest rate applied to both savings and GBM drift.
        sigma : float
            Volatility of the risky asset returns.
        instantaneous_utility : callable
            Function of daily consumption returning instantaneous utility.
        terminal_penalty : callable
            Function of terminal wealth returning penalty.
        """
        self.as_of_date: pd.Timestamp = as_of_date
        self.simulation_end_date: pd.Timestamp = simulation_end_date
        self.retirement_date: pd.Timestamp = retirement_date
        self.monthly_salary: float = monthly_salary
        self.monthly_pension: float = monthly_pension
        self.monthly_rent: float = monthly_rent
        self.r: float = r
        self.sigma: float = sigma
        self.instantaneous_utility: Callable[[float], float] = instantaneous_utility
        self.terminal_penalty: Callable[[float], float] = terminal_penalty

        # --- create daily time grid ---
        self.simulation_days: pd.DatetimeIndex = pd.date_range(
            start=as_of_date, end=simulation_end_date, freq="D"
        )
        self.n_days: int = len(self.simulation_days)

        # --- create GBM model internally ---
        self.gbm_model: GBMReturns = GBMReturns(
            as_of_date=self.as_of_date,
            simulation_days=self.simulation_days,
            r=self.r,
            sigma=self.sigma,
        )

        # --- placeholders ---
        self.n_sims: int = None
        self.svgs: pd.DataFrame = None
        self.invs: pd.DataFrame = None
        self.cnst: pd.DataFrame = None
        self.u: pd.DataFrame = None
        self.v: pd.DataFrame = None
        self.return_simulations: pd.DataFrame = None
        self.cashflows: pd.Series = self._compute_daily_cashflows()

    def _compute_daily_cashflows(self) -> pd.Series:
        """
        Compute daily net cashflows from salary/pension and rent.

        Returns
        -------
        pd.Series
            Daily cashflows indexed by simulation_days.
        """
        cashflows = np.zeros(self.n_days)
        for i, day in enumerate(self.simulation_days):
            inflow = self.monthly_salary if day < self.retirement_date else self.monthly_pension
            inflow -= self.monthly_rent
            cashflows[i] = inflow / 30.0  # daily conversion
        return pd.Series(cashflows, index=self.simulation_days, name="cashflows")

    def initialize_paths(self, n_sims: int):
        """
        Initialize state variables and GBM return simulations.

        Parameters
        ----------
        n_sims : int
            Number of Monte Carlo simulation paths.
        """
        self.n_sims = n_sims
        columns = [f"sim_{i}" for i in range(n_sims)]

        self.svgs = pd.DataFrame(0.0, index=self.simulation_days, columns=columns)
        self.invs = pd.DataFrame(0.0, index=self.simulation_days, columns=columns)
        self.cnst = pd.DataFrame(0.0, index=self.simulation_days, columns=columns)
        self.u = pd.DataFrame(0.0, index=self.simulation_days, columns=columns)
        self.v = pd.DataFrame(0.0, index=self.simulation_days, columns=columns)

        self.return_simulations = self.gbm_model.simulate(n_sims)

    def simulate_dynamics(self):
        """
        Forward simulate savings and investment dynamics given current controls.
        """
        for t in range(1, self.n_days):
            day = self.simulation_days[t]
            prev_day = self.simulation_days[t - 1]

            self.svgs.loc[day] = (
                self.svgs.loc[prev_day]
                + self.r * self.svgs.loc[prev_day] * self.gbm_model.dt
                + self.cashflows.loc[prev_day]
                - self.cnst.loc[prev_day]
                - self.u.loc[prev_day]
                + self.v.loc[prev_day]
            )
            self.invs.loc[day] = (
                self.invs.loc[prev_day] * self.return_simulations.loc[prev_day]
                + self.u.loc[prev_day]
                - self.v.loc[prev_day]
            )

    def total_expected_utility(self) -> float:
        """
        Compute total expected utility: sum of daily utilities minus terminal penalty.

        Returns
        -------
        float
            Mean expected utility across all simulation paths.
        """
        # Elementwise apply instantaneous utility to consumption
        utility_sum = self.cnst.map(self.instantaneous_utility).sum()

        terminal_wealth = self.svgs.iloc[-1] + self.invs.iloc[-1]
        penalty = terminal_wealth.map(self.terminal_penalty)

        return float((utility_sum - penalty).mean())


if __name__ == "__main__":
    portfolio_sim = PortfolioSimulation(
        as_of_date=pd.Timestamp("2025-01-01"),
        simulation_end_date=pd.Timestamp("2025-12-31"),
        retirement_date=pd.Timestamp("2025-07-01"),
        monthly_salary=5000,
        monthly_pension=2000,
        monthly_rent=1500,
        r=0.01,
        sigma=0.15,
        instantaneous_utility=log_utility,
        terminal_penalty=linear_penalty,
    )

    portfolio_sim.initialize_paths(n_sims=1000)
    portfolio_sim.simulate_dynamics()
    print("Simulation days:", portfolio_sim.simulation_days)
    print("Expected utility:", portfolio_sim.total_expected_utility())
