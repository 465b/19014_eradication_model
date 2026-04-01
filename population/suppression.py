"""
Carrying-capacity suppression functions for the population model.

Each function receives the current density field and the total potential
incoming flux (growth recruits + far-field settlers + near-field incoming
spread) and returns a (ny, nx) suppression factor in [0, 1] that is
applied to all additive fluxes for that timestep.

Using the *predicted* density ``N + ΔN`` rather than just ``N`` ensures
that suppression accounts for what the density *would become* if all
incoming flux were accepted, rather than only the current state.

Concrete implementations
------------------------

LinearSuppression
    ``max(0, 1 - (N + ΔN) / K)``

    Linear decline to zero when predicted density reaches K.
    Equivalent to the logistic limiting factor applied to the
    *expected* post-growth density.  Guarantees that cells approaching
    K receive vanishingly little additional flux.

ExponentialSuppression
    ``exp(-(N + ΔN) / K)``

    Exponential decay of the suppression factor.  Always positive —
    some flux is always accepted regardless of density.  Useful when a
    hard zero at K is undesirable (e.g. stochastic or sparse
    populations).

Configuration
-------------
Selected via ``organism.carrying_capacity_suppression`` in the scenario
config (``"linear"`` or ``"exponential"``).  Only active when
``organism.carrying_capacity`` is also provided.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SuppressionFunction(ABC):
    """
    Abstract base for carrying-capacity suppression functions.

    A suppression function receives the current per-cell density ``N``
    (ny, nx) and the total potential incoming flux ``delta_N`` (ny, nx)
    for one timestep and returns a factor in [0, 1] by which all
    additive fluxes are multiplied before being applied to the
    population.

    Subclasses must implement :meth:`__call__`.  :meth:`from_config`
    provides a config-driven factory.
    """

    @abstractmethod
    def __call__(
        self,
        N: np.ndarray,
        delta_N: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the (ny, nx) suppression factor.

        Parameters
        ----------
        N : (ny, nx) float32 array
            Current total density per cell (before any additions).
        delta_N : (ny, nx) float32 array
            Total potential incoming flux per cell this timestep
            (growth + far-field settlers + near-field incoming spread).

        Returns
        -------
        factor : (ny, nx) float32 array
            Values in [0, 1].  Multiply all additive fluxes by this
            before applying them to the age structure.
        """

    @classmethod
    def from_config(cls, kind: str, K: float) -> "SuppressionFunction":
        """
        Factory: build a suppression function from a config string.

        Parameters
        ----------
        kind : str
            ``"linear"`` or ``"exponential"``.
        K : float
            Per-cell carrying capacity [ind/m²].
        """
        if kind == "linear":
            return LinearSuppression(K)
        if kind == "exponential":
            return ExponentialSuppression(K)
        raise ValueError(
            f"Unknown carrying_capacity_suppression {kind!r}. "
            "Must be 'linear' or 'exponential'."
        )


class LinearSuppression(SuppressionFunction):
    """
    ``factor = max(0, 1 - (N + ΔN) / K)``

    Linear decline to zero as predicted density approaches K.  No flux
    is accepted once ``N + ΔN ≥ K``.
    """

    def __init__(self, K: float) -> None:
        if K <= 0:
            raise ValueError("K must be > 0")
        self.K = K

    def __call__(self, N: np.ndarray, delta_N: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - (N + delta_N) / self.K, 0.0, 1.0).astype(np.float32)


class ExponentialSuppression(SuppressionFunction):
    """
    ``factor = exp(-(N + ΔN) / K)``

    Exponential decay — always positive, no hard zero at K.  Useful
    when a strict hard boundary is undesirable.
    """

    def __init__(self, K: float) -> None:
        if K <= 0:
            raise ValueError("K must be > 0")
        self.K = K

    def __call__(self, N: np.ndarray, delta_N: np.ndarray) -> np.ndarray:
        return np.exp(-(N + delta_N) / self.K).astype(np.float32)
