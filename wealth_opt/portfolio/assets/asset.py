from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any


class Asset(ABC):
    """
    Abstract base class representing a tradable financial asset.

    This class defines the interface for any asset that has a drift (expected return)
    and volatility (diffusion intensity), which are typically functions of time and
    the asset's state.

    Attributes
    ----------
    None directly stored; subclasses must implement:
        - mu(t, x): Drift function μ(t, x)
        - sigma(t, x): Volatility function σ(t, x)
    """

    @abstractmethod
    def mu(self, t: float, x: Any) -> float:
        """
        Compute the drift (expected return) of the asset at a given time and state.

        Parameters
        ----------
        t : float
            Current time.
        x : Any
            Current state or value of the asset.

        Returns
        -------
        float
            The drift (expected rate of return) μ(t, x) of the asset.
        """
        pass

    @abstractmethod
    def sigma(self, t: float, x: Any) -> float:
        """
        Compute the volatility (diffusion intensity) of the asset at a given time and state.

        Parameters
        ----------
        t : float
            Current time.
        x : Any
            Current state or value of the asset.

        Returns
        -------
        float
            The volatility σ(t, x) of the asset.
        """
        pass
