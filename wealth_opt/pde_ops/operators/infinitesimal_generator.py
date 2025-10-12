import numpy as np
from scipy.sparse import diags, csc_matrix
from typing import Union

from wealth_opt.portfolio.portfolio import Portfolio


class InfGen:
    """
    Infinitesimal generator (differential operator) for a given optimal_control.

    For a optimal_control with dynamics:
        dX_t = μ(t, X_t, u) dt + σ(t, X_t, u) dW_t

    The operator acts as:
        (A f)(t, x) = μ(t, x, u) ∂f/∂x + 0.5 σ(t, x, u)^2 ∂²f/∂x²
    """

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    def get(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        c: float = 0.0,
        bc: str = "neumann",
    ) -> csc_matrix:
        """
        Construct the discrete generator matrix A^u(t, x) for a 1D mesh.

        Parameters
        ----------
        t : float
            Time at which to evaluate μ, σ.
        x : np.ndarray
            1D array of spatial grid points.
        u : np.ndarray
            Control vector (optimal_control weights). Must match number of assets.
        c : float, optional
            Consumption rate (used in μ).
        bc : str, optional
            Boundary condition ('dirichlet' or 'neumann').

        Returns
        -------
        csc_matrix
            Sparse finite-difference matrix representing A^u(t, x).
        """

        if len(u) != self.portfolio.n_assets:
            raise ValueError(
                f"Control vector u must have length {self.portfolio.n_assets}, "
                f"but got {len(u)}"
            )

        n = len(x)
        dx = np.diff(x)
        if not np.allclose(dx, dx[0]):
            raise ValueError("Grid x must be uniform for this simple scheme.")
        dx = dx[0]

        # Compute local coefficients
        mu_vals = np.array([self.portfolio.mu(t, xi, u, c) for xi in x])
        sigma_vals = np.array([self.portfolio.sigma(t, xi, u) for xi in x])
        diff_coeff = 0.5 * sigma_vals**2

        # --- Finite differences ---
        main_diag = np.zeros(n)
        upper_diag = np.zeros(n - 1)
        lower_diag = np.zeros(n - 1)

        for i in range(1, n - 1):
            mu = mu_vals[i]
            a = diff_coeff[i]

            lower_diag[i - 1] += a / dx**2 - mu / (2 * dx)
            main_diag[i]      += -2 * a / dx**2
            upper_diag[i]     += a / dx**2 + mu / (2 * dx)

        # --- Boundary conditions ---
        if bc == "dirichlet":
            main_diag[0] = main_diag[-1] = 1.0
        elif bc == "neumann":
            main_diag[0] = main_diag[-1] = -2 * diff_coeff[0] / dx**2
            upper_diag[0] = 2 * diff_coeff[0] / dx**2
            lower_diag[-1] = 2 * diff_coeff[-1] / dx**2
        else:
            raise ValueError("Unknown boundary condition type.")

        diagonals = [lower_diag, main_diag, upper_diag]
        A_mat = diags(diagonals, offsets=[-1, 0, 1], format="csc")

        return A_mat
