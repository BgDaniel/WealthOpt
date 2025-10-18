import numpy as np
from typing import List

from wealth_opt.portfolio.portfolio import Portfolio
from wealth_opt.portfolio.assets.savings_account import SavingsAccount
from wealth_opt.portfolio.assets.stock import Stock


class SPortfolio(Portfolio):
    """
    Concrete portfolio consisting of a single risky asset (stock)
    and a risk-free savings account.

    This portfolio is the simplest non-trivial example:
        - Asset 0: SavingsAccount (risk-free)
        - Asset 1: Stock (risky)

    The drift and volatility are computed using the inherited
    Portfolio formulas.

    Attributes
    ----------
    savings : SavingsAccount
        The risk-free asset.
    stock : Stock
        The risky asset.
    assets : List[Asset]
        List of the two assets (savings + stock).
    """

    def __init__(self, r: float, sigma: float) -> None:
        """
        Initialize a portfolio with one risk-free and one risky asset.

        Parameters
        ----------
        r : float
            Constant risk-free interest rate.
        sigma : float
            Volatility of the risky asset.
        """
        self.savings = SavingsAccount(r=r)
        self.stock = Stock(r=r, sigma=sigma)
        assets: List = [self.savings, self.stock]
        super().__init__(assets)