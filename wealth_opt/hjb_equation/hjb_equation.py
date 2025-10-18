import numpy as np
from typing import Callable, Dict, List
from scipy.sparse import csc_matrix

from wealth_opt.pde_ops.operators.infinitesimal_generator import InfGen
from wealth_opt.pde_ops.pde_solver.crank_nicolson import crank_nicolson



class HJBEquation:
    """
    Hamilton–Jacobi–Bellman (HJB) equation solver using Crank–Nicolson
    for PDE steps and policy iteration for portfolio controls.

    Solves:
        dV/dt + sup_u [ f(t, x, u) + A^u V(t, x) ] = 0
    with terminal condition V(T, x) = phi(x).

    Attributes
    ----------
    t : np.ndarray
        1D array of time grid points.
    x : np.ndarray
        1D spatial grid for wealth.
    n_x : int
        Number of spatial points.
    v : np.ndarray
        Value function array with shape (len(t), n_x).
    u_0 : List[np.ndarray]
        Initial control guess per asset.
    c_0 : np.ndarray
        Initial consumption or auxiliary control (if any).
    utility : Callable
        Instantaneous reward function.
    penalty : Callable
        Terminal wealth penalty function.
    inf_gen : InfGen
        Infinitesimal generator providing A^u matrices.
    n_assets : int
        Number of assets in the portfolio.
    cashflows : Dict[int, np.ndarray]
        Deterministic cashflows indexed by time-step.
    x_target : float
        Target wealth at terminal time.
    dx : float
        Grid spacing in wealth.
    x_max : float
        Maximum wealth for grid.
    """

    def __init__(
        self,
        t: np.ndarray,
        utility: Callable[[float, np.ndarray, Controls], np.ndarray],
        penalty: Callable[[np.ndarray, float], np.ndarray],
        inf_gen: InfGen,
        n_assets: int,
        cashflows: Dict[int, np.ndarray],
        x_target: float = 0.0,
        dx: float = 10.0,
        x_max: float = 100000.0,
    ) -> None:
        self.t: np.ndarray = np.asarray(t)
        self.n_times: int = len(self.t)

        self.utility: Callable[[float, np.ndarray, Controls], np.ndarray] = utility
        self.penalty: Callable[[np.ndarray, float], np.ndarray] = penalty
        self.inf_gen: InfGen = inf_gen
        self.n_assets: int = n_assets
        self.cashflows: Dict[int, np.ndarray] = cashflows
        self.x_target: float = x_target

        self.dx: float = dx
        self.x_max: float = x_max

        self.x: np.ndarray = np.arange(0.0, self.x_max, self.dx)
        self.n_x: int = len(self.x)

        self.v: np.ndarray = np.zeros((self.n_times, self.n_x))

        # Initial guess for control (equal weights)
        self.u_0: List[np.ndarray] = [
            np.full(self.n_x, 1.0 / self.n_assets) for _ in range(self.n_assets)
        ]
        self.c_0: np.ndarray = np.zeros(self.n_x)

    # -------------------------------------------------------------------------
    def solve(
        self,
        tol_v: float = 1e-6,
        tol_u: float = 1e-5,
        max_policy_iter: int = 20,
        max_pde_iter: int = 100,
    ) -> np.ndarray:
        """
        Solve the HJB equation via backward time-stepping with
        Crank–Nicolson PDE steps and policy iteration.

        Parameters
        ----------
        tol_v : float, optional
            Tolerance for value function convergence.
        tol_u : float, optional
            Tolerance for policy (control) convergence.
        max_policy_iter : int, optional
            Maximum iterations for control improvement per time step.
        max_pde_iter : int, optional
            Maximum iterations for PDE solver (unused here but for extensibility).

        Returns
        -------
        np.ndarray
            Value function array v(t, x) of shape (len(t), n_x).
        """
        # Terminal condition at final time
        self.v[-1, ...] = self.penalty(self.x, target=self.x_target)

        # Backward time-stepping
        for n in range(self.n_times - 2, -1, -1):
            t_m, t_n = self.t[n], self.t[n + 1]
            dt: float = t_n - t_m

            v_next: np.ndarray = self.v[n + 1, ...]
            v_current: np.ndarray = v_next.copy()

            # Initialize control
            u_current: List[np.ndarray] = self.u_0
            c_current: np.ndarray = self.c_0

            for policy_iter in range(max_policy_iter):
                cap_a: csc_matrix = self.inf_gen.matrix(t_m, self.x, u_current)

                # PDE step (Crank–Nicolson)
                v_new: np.ndarray = crank_nicolson(v_next, u_current, dt, cap_a)

                # Policy improvement step
                u_new: List[np.ndarray] = self._update_optimal_controls(t_m, v_new)

                # Check convergence of control
                du: float = np.max(
                    [
                        np.nanmax(np.abs(mesh_new - mesh_old))
                        for mesh_new, mesh_old in zip(u_new, u_current)
                    ]
                )
                if du < tol_u:
                    u_current = u_new
                    v_current = v_new
                    break

                u_current = u_new
                v_current = v_new
            else:
                print(f"Warning: Control iteration did not converge at time index {n}")

            self.v[n, ...] = v_current

        return self.v

    def _update_optimal_controls(self, t_n: float, v_n: np.ndarray) -> Coords:
        """
        Compute optimal control u*(x) = argmax_u [ f + A^u V_n ].

        Parameters
        ----------
        t_n : float
            Current time.
        v_n : np.ndarray
            Current value function slice.

        Returns
        -------
        Coords
            Optimal control coordinate mesh (Δu_1,...,Δu_n,c).
        """
        # Example: use uniform weights and mean asset values as reference
        current_u = np.ones(self.u.dim) / self.u.dim
        current_S = np.mean([mesh for mesh in self.states], axis=0)

        # Get admissible controls as mesh
        valid_controls = self.u.admissible_controls(current_u, current_S)

        # Debug print
        print("Admissible control meshes:")
        for i, mesh in enumerate(valid_controls):
            print(f"Control dimension {i}: shape = {mesh.shape}, values = \n{mesh}")

        # Evaluate H = f + A^u V for each control
        h_values = []
        for u_val_idx in np.ndindex(valid_controls[0].shape):
            u_val = np.array([mesh[u_val_idx] for mesh in valid_controls])
            h = self.utility(t_n, self.states, u_val)
            if self.a is not None:
                h += self.a.construct_generator_matrix(v_n, t_n, self.states, u_val)
            h_values.append(h)

        h_stack = np.stack(h_values, axis=0)  # shape = (n_controls, *spatial_shape)
        u_star_idx = np.argmax(h_stack, axis=0)

        # Select optimal control elementwise
        u_stack = np.stack(valid_controls, axis=0)  # shape = (n_assets+1, *mesh_shape)
        u_star = np.take(u_stack, u_star_idx, axis=0)
        return u_star
