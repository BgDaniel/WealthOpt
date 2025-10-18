import pandas as pd
from typing import Callable, Dict

from wealth_opt.hjb_equation.hjb_equation import HJBEquation
from wealth_opt.pde_ops.operators.infinitesimal_generator import InfGen
from wealth_opt.portfolio.assets.savings_account import SavingsAccount
from wealth_opt.portfolio.assets.stock import Stock
from wealth_opt.portfolio.portfolio import Portfolio
from wealth_opt.utilities import log_utility, crra
from wealth_opt.penalties import quadratic_penalty


class OptimalControl:
    """
    Simulation environment for portfolio optimization.

    Holds all parameters, time grids, deterministic cashflows, and
    the infinitesimal generator needed for solving HJB equations.
    Does not perform optimization itself.
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
        terminal_target_wealth: float = 0.0,
    ) -> None:
        """
        Initialize the simulation environment.

        Parameters
        ----------
        as_of_date : pd.Timestamp
            Simulation start date.
        simulation_end_date : pd.Timestamp
            Simulation end date.
        retirement_date : pd.Timestamp
            Date when salary stops and pension starts.
        monthly_salary : float
            Salary before retirement (per month).
        monthly_pension : float
            Pension after retirement (per month).
        monthly_rent : float
            Monthly rent outflow (negative cashflow).
        r : float
            Risk-free rate for savings account and drift of risky asset.
        sigma : float
            Volatility of risky asset returns.
        utility : Callable[[float], float], optional
            Instantaneous utility function of consumption (default: CRRA).
        penalty : Callable[[float], float], optional
            Terminal wealth penalty function (default: quadratic).
        terminal_target_wealth : float, optional
            Target wealth at the terminal time (default: 0.0).
        """
        # --- Dates & time grids ---
        self.as_of_date: pd.Timestamp = as_of_date
        self.simulation_end_date: pd.Timestamp = simulation_end_date
        self.retirement_date: pd.Timestamp = retirement_date

        self.simulation_days: pd.DatetimeIndex = pd.date_range(
            start=as_of_date, end=simulation_end_date, freq="D"
        )
        self.n_days: int = len(self.simulation_days)
        self.t: pd.Series = (self.simulation_days - self.as_of_date).days / 365.25

        self.monthly_grid: pd.DatetimeIndex = pd.date_range(
            start=as_of_date, end=simulation_end_date, freq="M"
        )

        # --- Cashflows ---
        self.monthly_salary: float = monthly_salary
        self.monthly_pension: float = monthly_pension
        self.monthly_rent: float = monthly_rent

        self.salary_cf: pd.Series = self._rollout_salary()
        self.rent_cf: pd.Series = self._rollout_rent()
        self.deterministic_cashflows: Dict[int, float] = self._deterministic_cashflows()

        # --- Assets & portfolio ---
        self.r: float = r
        self.sigma: float = sigma

        self.savings: SavingsAccount = SavingsAccount(r=r)
        self.stock: Stock = Stock(r=r, sigma=sigma)
        self.portfolio: Portfolio = Portfolio([self.savings, self.stock])

        self.utility: Callable[[float], float] = utility
        self.penalty: Callable[[float], float] = penalty
        self.terminal_target_wealth: float = terminal_target_wealth

        # --- Infinitesimal generator ---
        self.inf_gen: InfGen = InfGen(portfolio=self.portfolio)

    # -------------------------------------------------------------------------
    def _rollout_salary(self) -> pd.Series:
        """Roll out salary/pension over the monthly grid."""
        inflow = [
            self.monthly_salary if date < self.retirement_date else self.monthly_pension
            for date in self.monthly_grid
        ]
        return pd.Series(inflow, index=self.monthly_grid, name="salary_pension_cf")

    # -------------------------------------------------------------------------
    def _rollout_rent(self) -> pd.Series:
        """Roll out constant negative rent over the monthly grid."""
        rent_outflow = [-self.monthly_rent] * len(self.monthly_grid)
        return pd.Series(rent_outflow, index=self.monthly_grid, name="rent_cf")

    # -------------------------------------------------------------------------
    def _deterministic_cashflows(self) -> Dict[int, float]:
        """
        Combine salary/pension and rent into a daily-indexed cashflow dict.

        Returns
        -------
        Dict[int, float]
            Keys: integer index of simulation day.
            Values: net deterministic cashflow at that day.
        """
        net_cf = self.salary_cf + self.rent_cf
        cashflow_dict: Dict[int, float] = {}

        for date, value in net_cf.items():
            idx = (abs(self.simulation_days - date)).argmin()
            cashflow_dict[idx] = cashflow_dict.get(idx, 0.0) + value

        return cashflow_dict

    # -------------------------------------------------------------------------
    def determine_opt_control(self) -> None:
        """
        Solve the HJB equation for the given deterministic cashflows,
        portfolio, and utility/penalty specification.
        """
        hjb_equation = HJBEquation(
            t=self.t,
            utility=self.utility,
            penalty=self.penalty,
            inf_gen=self.inf_gen,
            n_assets=self.portfolio.n_assets,
            cashflows=self.deterministic_cashflows,
            x_target=self.terminal_target_wealth,
        )
        hjb_equation.solve()


# -------------------------------------------------------------------------
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
        penalty=quadratic_penalty,
    )

    optimal_control.determine_opt_control()
