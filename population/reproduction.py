"""
Growth / recruitment models for the population simulation.

Each model receives the current total density field and returns a
``(ny, nx)`` array of *new recruits* to be placed into the youngest
age bin.  The growth model does **not** modify the existing density
in-place — the caller (:class:`PopulationModel`) is responsible for
feeding recruits into the age structure.

Two implementations are provided:

    LogisticGrowth  — density-dependent recruitment bounded by a carrying
                      capacity *K*.
    ExponentialGrowth — unbounded constant-rate recruitment (useful for
                        testing / short-horizon runs).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class GrowthModel(ABC):
    """Abstract base for population growth / recruitment models."""

    def __init__(self) -> None:
        self._log: list[dict[str, Any]] = []

    @abstractmethod
    def step(
        self,
        total_density: np.ndarray,
        habitat_mask: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        """
        Compute the recruitment field for one model timestep.

        Parameters
        ----------
        total_density : (ny, nx) float32 array
            Total population density summed across all age bins.
        habitat_mask : (ny, nx) bool array
            True = suitable habitat.
        timestep : int
            Current model timestep index (0-based).

        Returns
        -------
        recruits : (ny, nx) float32 array
            New individuals to add to age-bin 0.  Non-negative.
        """

    @property
    def log(self) -> list[dict[str, Any]]:
        return self._log

    @classmethod
    def from_config(cls, organism_cfg: dict) -> "GrowthModel":
        """
        Factory: dispatch on ``organism_cfg["growth_model"]``.

        Supported values:
            ``"logistic"``     → :class:`LogisticGrowth`
            ``"exponential"``  → :class:`ExponentialGrowth`

        Falls back to ``"logistic"`` if the key is absent.
        """
        kind = organism_cfg.get("growth_model", "logistic")

        if kind == "logistic":
            return LogisticGrowth(
                growth_rate=float(organism_cfg["growth_rate_per_week"]),
                carrying_capacity=float(organism_cfg["carrying_capacity"]),
            )

        if kind == "exponential":
            return ExponentialGrowth(
                growth_rate=float(organism_cfg["growth_rate_per_week"]),
            )

        raise ValueError(
            f"Unknown growth_model {kind!r}. "
            "Must be 'logistic' or 'exponential'."
        )


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class LogisticGrowth(GrowthModel):
    """
    Logistic recruitment: new individuals appear at a rate that slows as
    total density approaches the carrying capacity *K*.

    .. math::

        \\text{recruits} = r \\cdot N \\cdot \\left(1 - \\frac{N}{K}\\right)

    where *N* is the total density per cell, *r* is the intrinsic growth
    rate per week, and *K* is the per-cell carrying capacity.  Recruitment
    is zero in unsuitable habitat and clamped to non-negative.

    Parameters
    ----------
    growth_rate : float
        Intrinsic per-week growth rate *r*.
    carrying_capacity : float
        Per-cell carrying capacity *K* (individuals/m²).
    """

    def __init__(self, growth_rate: float, carrying_capacity: float) -> None:
        super().__init__()
        if carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be > 0")
        self.r = growth_rate
        self.K = carrying_capacity

    def step(
        self,
        total_density: np.ndarray,
        habitat_mask: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        recruits = self.r * total_density * (1.0 - total_density / self.K)
        recruits[~habitat_mask] = 0.0
        np.clip(recruits, 0.0, None, out=recruits)
        recruits = recruits.astype(np.float32)

        self._log.append({
            "timestep": timestep,
            "total_recruits": float(recruits.sum()),
        })
        return recruits


class ExponentialGrowth(GrowthModel):
    """
    Exponential (unbounded) recruitment.

    .. math::

        \\text{recruits} = r \\cdot N

    Useful for short-horizon test runs or when external mortality keeps
    the population in check.

    Parameters
    ----------
    growth_rate : float
        Intrinsic per-week growth rate *r*.
    """

    def __init__(self, growth_rate: float) -> None:
        super().__init__()
        self.r = growth_rate

    def step(
        self,
        total_density: np.ndarray,
        habitat_mask: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        recruits = self.r * total_density
        recruits[~habitat_mask] = 0.0
        np.clip(recruits, 0.0, None, out=recruits)
        recruits = recruits.astype(np.float32)

        self._log.append({
            "timestep": timestep,
            "total_recruits": float(recruits.sum()),
        })
        return recruits
