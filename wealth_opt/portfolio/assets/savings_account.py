from wealth_opt.portfolio.assets.asset import Asset


class SavingsAccount(Asset):
    """
    Deterministic savings account with a constant risk-free rate.

    The dynamics are given by:
        dB_t = r * B_t * dt

    Attributes
    ----------
    r : float
        Constant risk-free interest rate.
    """

    def __init__(self, r: float) -> None:
        """
        Initialize the savings account.

        Parameters
        ----------
        r : float
            Constant risk-free interest rate.
        """
        self._r = r

    # --- Drift (μ) ---
    def mu(self, t: float, x: float) -> float:
        """
        Return the drift of the savings account at a given time and state.

        For a deterministic savings account, the drift is constant:
            μ(t, B_t) = r

        Parameters
        ----------
        t : float
            Current time (not used in this deterministic case).
        x : float
            Current account balance (not used in this deterministic case).

        Returns
        -------
        float
            Constant drift r.
        """
        return self._r

    # --- Volatility (σ) ---
    def sigma(self, t: float, x: float) -> float:
        """
        Return the volatility of the savings account.

        For a deterministic account, volatility is zero:
            σ(t, B_t) = 0

        Parameters
        ----------
        t : float
            Current time (not used).
        x : float
            Current account balance (not used).

        Returns
        -------
        float
            Zero volatility.
        """
        return 0.0
