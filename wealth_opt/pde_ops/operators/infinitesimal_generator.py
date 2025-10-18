import numpy as np
from scipy.sparse import diags, csc_matrix
from typing import Literal

from wealth_opt.portfolio.portfolio import Portfolio


class InfGen:
    """
    Infinitesimal generator (differential operator) for a given portfolio control.

    For a controlled SDE of the form:
        dX_t = μ(t, X_t, u) dt + σ(t, X_t, u) dW_t

    The infinitesimal generator acts as:
        (A f)(t, x) = μ(t, x, u) ∂f/∂x + 0.5 σ(t, x, u)^2 ∂²f/∂x²

    Attributes
    ----------
    portfolio : Portfolio
        Portfolio object providing drift μ and volatility σ methods.
    """

    def __init__(self, portfolio: Portfolio) -> None:
        """
        Initialize the infinitesimal generator for a given portfolio.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio object containing assets and their dynamics.
        """
        self.portfolio: Portfolio = portfolio

    # -------------------------------------------------------------------------
    def matrix(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        c: float = 0.0,
        bc: Literal["dirichlet", "neumann"] = "neumann",
    ) -> csc_matrix:
        """
        Construct the finite-difference approximation of the generator matrix A^u(t, x)
        for a 1D spatial mesh.

        Parameters
        ----------
        t : float
            Time at which to evaluate μ and σ.
        x : np.ndarray
            1D array of spatial grid points (must be uniform).
        u : np.ndarray
            Control array of shape (n_assets, len(x)), specifying the portfolio weights
            or control values for each asset at each spatial point.
        c : float, optional
            Consumption or additional drift term (default: 0.0).
        bc : {'dirichlet', 'neumann'}, optional
            Boundary condition type:
                - 'dirichlet' : fixed value at boundaries.
                - 'neumann'   : zero-gradient (reflecting) at boundaries.

        Returns
        -------
        csc_matrix
            Sparse CSC matrix representing the discrete infinitesimal generator
            for the given control and grid.
        """
        n: int = len(x)
        dx: float = np.diff(x)[0]
        if not np.allclose(np.diff(x), dx):
            raise ValueError(
                "Grid x must be uniform for this finite-difference scheme."
            )

        # Validate control array shape
        if len(u) != self.portfolio.n_assets:
            raise ValueError(
                f"Control array u must have length equal to number of assets "
                f"({self.portfolio.n_assets}), but got {len(u)}."
            )
        if not all(u[i].shape == x.shape for i in range(self.portfolio.n_assets)):
            raise ValueError("Each u[i] must have the same shape as x.")

        # Compute drift and diffusion coefficients
        mu_vals: np.ndarray = self.portfolio.mu(t, x, u, c)
        sigma_vals: np.ndarray = self.portfolio.sigma(t, x, u)
        diff_coeff: np.ndarray = 0.5 * sigma_vals**2

        # Initialize finite-difference diagonals
        main_diag: np.ndarray = np.zeros(n)
        upper_diag: np.ndarray = np.zeros(n - 1)
        lower_diag: np.ndarray = np.zeros(n - 1)

        for i in range(1, n - 1):
            mu = mu_vals[i]
            a = diff_coeff[i]
            lower_diag[i - 1] += a / dx**2 - mu / (2 * dx)
            main_diag[i] += -2 * a / dx**2
            upper_diag[i] += a / dx**2 + mu / (2 * dx)

        # Apply boundary conditions
        if bc == "dirichlet":
            main_diag[0] = main_diag[-1] = 1.0
        elif bc == "neumann":
            main_diag[0] = main_diag[-1] = -2 * diff_coeff[0] / dx**2
            upper_diag[0] = 2 * diff_coeff[0] / dx**2
            lower_diag[-1] = 2 * diff_coeff[-1] / dx**2
        else:
            raise ValueError(f"Unknown boundary condition type: {bc}")

        # Assemble sparse matrix
        diagonals = [lower_diag, main_diag, upper_diag]
        return diags(diagonals, offsets=[-1, 0, 1], format="csc")
