import numpy as np
from scipy.sparse import identity, csc_matrix
from scipy.sparse.linalg import spsolve


def crank_nicolson(
    v_next: np.ndarray, u: np.ndarray, dt: float, cap_a: csc_matrix
) -> np.ndarray:
    """
    Perform one Crank–Nicolson step for a linear PDE using the next value v_{n+1}.

    Solves:
        (I - 0.5 * dt * A) v_n = (I + 0.5 * dt * A) v_{n+1}

    Parameters
    ----------
    v_next : np.ndarray
        Value function at the next time step v_{n+1}.
    u : np.ndarray
        Control vector or matrix (not directly used here, but kept for interface consistency).
    dt : float
        Time step size.
    cap_a : csc_matrix
        Infinitesimal generator matrix A^u(t, x) in CSC format.

    Returns
    -------
    np.ndarray
        Value function at the current time step v_n (same shape as v_next).
    """
    if not isinstance(cap_a, csc_matrix):
        raise TypeError("cap_a must be a scipy.sparse.csc_matrix")

    # Identity matrix of same dimension as A
    I = identity(cap_a.shape[0], format="csc")

    # Build LHS and RHS operators
    LHS = (I - 0.5 * dt * cap_a).tocsc()
    RHS = (I + 0.5 * dt * cap_a).tocsc()

    # Solve linear system: LHS * v_n = RHS * v_{n+1}
    v_n_flat = spsolve(LHS, RHS @ v_next)

    # Reshape back to original grid shape
    return v_n_flat.reshape(v_next.shape)
