import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

from wealth_opt.pde_ops.operators.infinitesimal_generator import InfGen


def crank_nicolson_step(
    v_next: np.ndarray,
    inf_gen: InfGen,
    u: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Perform one Crank–Nicolson step for a linear PDE using the next value v_{n+1}.

    Solves: (I - 0.5*dt*A) v_n = (I + 0.5*dt*A) v_{n+1}

    Parameters
    ----------
    v_next : np.ndarray
        Value function at next time step v_{n+1}.
    inf_gen : object
        Object providing method `matrix(u)` returning sparse PDE matrix.
    u : np.ndarray
        Control at current step.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        Value function at current time step v_n (same shape as v_next).
    """
    # PDE matrix for current step
    A = inf_gen.matrix(u)

    # Flatten for sparse solve
    v_next_flat = v_next.ravel()

    # LHS and RHS operators
    I = identity(A.shape[0], format="csc")
    LHS = (I - 0.5 * dt * A).tocsc()
    RHS = (I + 0.5 * dt * A).tocsc()

    # Solve linear system
    v_n_flat = spsolve(LHS, RHS.dot(v_next_flat))

    return v_n_flat.reshape(v_next.shape)