import numpy as np
import pandas as pd
from typing import Callable

from wealth_opt.environment.portfolio_controls import PortfolioControls
from wealth_opt.environment.portfolio_states import PortfolioStates
from wealth_opt.hjb_equation.hjb_equation import HJBEquation
from wealth_opt.pde_ops.operators.infinitesimal_generator import InfGen
from wealth_opt.portfolio.assets.asset import Asset
from wealth_opt.portfolio.assets.savings_account import SavingsAccount
from wealth_opt.portfolio.assets.stock import Stock
from wealth_opt.portfolio.portfolio import Portfolio
from wealth_opt.utilities import log_utility, crra
from wealth_opt.penalties import quadratic_penalty, linear_penalty
from wealth_opt.market_simulation.stock_returns import StockReturns


class OptimalControl:
    """
    Holds all parameters, time grids, cashflows, and GBM simulations
    for a optimal_control simulation environment. No optimization yet.
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
        utility: Callable[[float], float] = crra,
        penalty: Callable[[float], float] = quadratic_penalty,
    ):
        """
        Initialize the optimal_control simulation environment.

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
            Volatility of the risky assets returns.
        utility : callable
            Function of daily consumption returning instantaneous utility.
        penalty : callable
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

        self.savings = SavingsAccount(r=r)
        self.stock = Stock(r=r, sigma=sigma)
        self.portfolio = Portfolio([self.savings, self.stock])

        self.utility: Callable[[float], float] = utility
        self.penalty: Callable[[float], float] = penalty

        # --- create daily time grid ---
        self.simulation_days: pd.DatetimeIndex = pd.date_range(
            start=as_of_date, end=simulation_end_date, freq="D"
        )
        self.n_days: int = len(self.simulation_days)

        self.t = (self.simulation_days - self.as_of_date).days / 365.25

        self.states = PortfolioStates(n_assets=self.portfolio.n_assets)
        self.x = self.states.x

        self.controls = PortfolioControls(n_assets=self.portfolio.n_assets)

        self.inf_gen = InfGen(portfolio=self.portfolio)

    def determine_opt_control(self):


        hjb_equation = HJBEquation(
            t= self.t,
            x=self.x,
            u= self.controls,
            f=self.utility,
            phi=self.penalty,
            inf_gen=self.inf_gen
        )

        hjb_equation.solve()



if __name__ == "__main__":
    optimal_control = OptimalControl(
        as_of_date=pd.Timestamp("2025-01-01"),
        simulation_end_date=pd.Timestamp("2025-12-31"),
        retirement_date=pd.Timestamp("2025-07-01"),
        monthly_salary=5000,
        monthly_pension=2000,
        monthly_rent=1500,
        r=0.01,
        sigma=0.15,
        utility=log_utility,
        penalty=linear_penalty,
    )

    optimal_control.determine_opt_control()


