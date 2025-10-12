from wealth_opt.portfolio.assets.asset import Asset


class Stock(Asset):
    """
    Risky asset following Geometric Brownian Motion (GBM).

    The dynamics are given by:
        dS_t = μ(t, S_t) * S_t * dt + σ(t, S_t) * S_t * dW_t

    Attributes
    ----------
    r : float
        Constant drift (expected return).
    sigma_const : float
        Constant volatility.
    """

    def __init__(self, r: float, sigma: float) -> None:
        """
        Initialize a GBM stock asset.

        Parameters
        ----------
        r : float
            Constant drift (expected return).
        sigma : float
            Constant volatility.
        """
        self._r = r
        self._sigma = sigma

    # --- Drift (μ) ---
    def mu(self, t: float, x: float) -> float:
        """
        Compute the drift of the stock at a given time and state.

        For a GBM stock, the drift is adjusted by the variance term:
            μ_GBM(t, S_t) = (r - σ^2 / 2) * t

        Parameters
        ----------
        t : float
            Current time.
        x : float
            Current stock price (not used in constant-drift GBM).

        Returns
        -------
        float
            Drift at time t.
        """
        return (self._r - self._sigma**2 / 2.0) * t

    # --- Volatility (σ) ---
    def sigma(self, t: float, x: float) -> float:
        """
        Return the constant volatility of the stock.

        Parameters
        ----------
        t : float
            Current time (not used for constant volatility).
        x : float
            Current stock price (not used for constant volatility).

        Returns
        -------
        float
            Constant volatility σ.
        """
        return self._sigma
