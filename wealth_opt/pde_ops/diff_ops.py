from typing import Callable
import numpy as np

# --- Type aliases ---
F = Callable[[np.ndarray], float]  # f: ℝⁿ → ℝ (scalar-valued)
EPS = 1e-3  # minimum mesh spacing


class Differential:
    """Represents a scalar function f(x) and allows finite-difference differentiation."""

    def __init__(self, f: F):
        self.f = f

    def div(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Compute first derivative of f along axis i using finite differences."""
        x = np.asarray(x)

        # Evaluate function on the mesh
        f_vals = np.vectorize(self.f)(x)

        # Check mesh spacing
        dx_vals = np.diff(x, axis=axis)
        if np.any(dx_vals < EPS):
            raise ValueError(f"Mesh spacing along axis {axis} smaller than EPS={EPS}")

        # Compute derivative
        dx_mean = np.mean(dx_vals)
        deriv = np.gradient(f_vals, dx_mean, axis=axis)

        # Boundary handling
        slicer = [slice(None)] * deriv.ndim
        slicer[axis] = 0
        deriv[tuple(slicer)] = np.nan
        slicer[axis] = -1
        deriv[tuple(slicer)] = np.nan

        return deriv

    def div2(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        """Compute second derivative of f along axes (i,j)."""
        x = np.asarray(x)
        f_vals = np.vectorize(self.f)(x)

        dx_i = np.mean(np.diff(x, axis=i))
        dx_j = np.mean(np.diff(x, axis=j))

        if dx_i < EPS or dx_j < EPS:
            raise ValueError(f"Mesh spacing too small along axes {i} or {j} (EPS={EPS})")

        if i == j:
            first = np.gradient(f_vals, dx_i, axis=i)
            second = np.gradient(first, dx_i, axis=i)
        else:
            first = np.gradient(f_vals, dx_i, axis=i)
            second = np.gradient(first, dx_j, axis=j)

        # Boundary to NaN
        for ax in (i, j):
            slicer = [slice(None)] * second.ndim
            for idx in (0, -1):
                slicer[ax] = idx
                second[tuple(slicer)] = np.nan

        return second

    def __truediv__(self, rhs):
        """Enable syntax: d(f)/d(x,i) or d2(f)/d2(x,i,j)."""
        if isinstance(rhs, D_Operator):
            return self.div(rhs.x, rhs.axis)
        elif isinstance(rhs, D2_Operator):
            return self.div2(rhs.x, rhs.i, rhs.j)
        else:
            raise TypeError("Right-hand side must be d(x,i) or d2(x,i,j)")


# --- Operator classes ---

class D_Operator:
    def __init__(self, x: np.ndarray, axis: int):
        self.x = x
        self.axis = axis


class D2_Operator:
    def __init__(self, x: np.ndarray, i: int, j: int):
        self.x = x
        self.i = i
        self.j = j


# --- Helper constructors ---

def d(f: F) -> Differential:
    """Wrap a scalar function f(x) for first derivative."""
    return Differential(f)


def d2(f: F) -> Differential:
    """Wrap a scalar function f(x) for second derivative."""
    return Differential(f)


def d_(x: np.ndarray, i: int) -> D_Operator:
    """First-order differential operator."""
    return D_Operator(x, i)


def d2_(x: np.ndarray, i: int, j: int) -> D2_Operator:
    """Second-order differential operator."""
    return D2_Operator(x, i, j)


# --- Example usage ---

if __name__ == "__main__":
    x = np.linspace(0, 2 * np.pi, 100)

    def f(x: np.ndarray) -> float:
        return float(np.sin(x))  # scalar-valued

    df_dx = d(f) / d_(x, 0)
    print("df/dx ≈ cos(x):", df_dx[:5])

    d2f_dx2 = d2(f) / d2_(x, 0, 0)
    print("d²f/dx² ≈ -sin(x):", d2f_dx2[:5])
