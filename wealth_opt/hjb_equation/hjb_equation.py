import numpy as np
from typing import Callable, List
from wealth_opt.environment.interfaces.controls import Controls
from wealth_opt.environment.interfaces.states import Coords
from wealth_opt.pde_ops.operators.infinitesimal_generator import InfGen
from wealth_opt.pde_ops.pde_solver.crank_nicolson import crank_nicolson_step


class HJBEquation:
    """
    General representation of a Hamilton–Jacobi–Bellman (HJB) equation.

    dV/dt + sup_u [ f(t, x, u) + A^u V(t, x) ] = 0
    with terminal condition V(T, x) = phi(x).

    Attributes
    ----------
    t : np.ndarray
        Time grid.
    x : Coords
        State space mesh (list of ndarrays).
    u : Controls
        Portfolio control object.
    f : Callable
        Reward function: f(t, x, u) -> ndarray.
    phi : Callable
        Terminal condition: phi(x) -> ndarray.
    v : np.ndarray
        Value function array with shape (len(t), *spatial_shape).
    """

    def __init__(
        self,
        t: np.ndarray,
        x: Coords,
        u: Controls,
        f: Callable,
        phi: Callable,
        inf_gen: InfGen
    ) -> None:
        self.t = np.asarray(t)
        self.n_times = len(self.t)

        self.x = x
        self.u = u
        self.f = f
        self.phi = phi
        self.ing_gen = inf_gen

        spatial_shape = self.x[0].shape  # shape of a single mesh dimension
        self.v = np.zeros((len(self.t),) + spatial_shape, dtype=float)

    def solve(
        self,
        tol_v: float = 1e-6,
        tol_u: float = 1e-5,
        max_policy_iter: int = 20,
        max_pde_iter: int = 100
    ) -> np.ndarray:
        """
        Solve the HJB using Crank–Nicolson in time and policy (u) iteration.

        Returns
        -------
        np.ndarray
            Value function array v(t, x).
        """
        # Terminal condition at t = T
        self.v[-1, ...] = self.phi(self.x)

        # Backward time-stepping
        for n in range(self.n_times - 2, -1, -1):
            t_n, t_np1 = self.t[n], self.t[n + 1]
            dt = t_np1 - t_n
            v_next = self.v[n + 1, ...]
            v_current = v_next.copy()

            # Initialize control (e.g., static/default control)
            u_current = self.u.static_controls()

            for policy_iter in range(max_policy_iter):
                inf_gen = self.inf_gen.get(t_n, self.x, u_current)

                # 1️⃣ PDE step (Crank–Nicolson) for fixed control
                v_new = crank_nicolson_step(
                    v_next, inf_gen, u_current, dt
                )

                # 2️⃣ Policy improvement step
                u_new = self._update_optimal_controls(t_n, v_new)

                # 3️⃣ Check control convergence
                du = np.max([np.nanmax(np.abs(mesh_new - mesh_old))
                             for mesh_new, mesh_old in zip(u_new, u_current)])
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

    def _update_optimal_controls(
        self,
        t_n: float,
        v_n: np.ndarray
    ) -> Coords:
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
        current_S = np.mean([mesh for mesh in self.x], axis=0)

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
            h = self.f(t_n, self.x, u_val)
            if self.a is not None:
                h += self.a.get(v_n, t_n, self.x, u_val)
            h_values.append(h)

        h_stack = np.stack(h_values, axis=0)  # shape = (n_controls, *spatial_shape)
        u_star_idx = np.argmax(h_stack, axis=0)

        # Select optimal control elementwise
        u_stack = np.stack(valid_controls, axis=0)  # shape = (n_assets+1, *mesh_shape)
        u_star = np.take(u_stack, u_star_idx, axis=0)
        return u_star
