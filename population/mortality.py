"""
Natural mortality models for the population simulation.

Natural mortality is applied *independently* of eradication-induced
mortality (which is handled by the culling module).  Each model receives
the full age-structured density and returns the updated density after
applying age- or environment-dependent die-off.

Two implementations are provided:

    NoMortality       — identity function (placeholder for v1).
    FlatSurvival      — constant per-week survival rate applied uniformly
                        across all age bins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class MortalityModel(ABC):
    """Abstract base for natural mortality models."""

    def __init__(self) -> None:
        self._log: list[dict[str, Any]] = []

    @abstractmethod
    def step(
        self,
        density: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        """
        Apply natural mortality to the age-structured density field.

        Parameters
        ----------
        density : (n_ages, ny, nx) float32 array
            Full age-structured density.
        timestep : int
            Current model timestep index (0-based).

        Returns
        -------
        density : (n_ages, ny, nx) float32 array
            Updated density after natural mortality.
        """

    @property
    def log(self) -> list[dict[str, Any]]:
        return self._log

    @classmethod
    def from_config(cls, organism_cfg: dict) -> "MortalityModel":
        """
        Factory: build a mortality model from the ``organism`` config.

        Dispatches on ``organism_cfg.get("mortality_model", "none")``:
            ``"none"``          → :class:`NoMortality`
            ``"flat_survival"`` → :class:`FlatSurvival`
        """
        kind = organism_cfg.get("mortality_model", "none")

        if kind == "none":
            return NoMortality()

        if kind == "flat_survival":
            return FlatSurvival(
                survival_per_week=float(organism_cfg["natural_survival_per_week"]),
            )

        raise ValueError(
            f"Unknown mortality_model {kind!r}. "
            "Must be 'none' or 'flat_survival'."
        )


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class NoMortality(MortalityModel):
    """No natural mortality — identity function."""

    def step(self, density: np.ndarray, timestep: int) -> np.ndarray:
        self._log.append({"timestep": timestep, "total_mortality": 0.0})
        return density


class FlatSurvival(MortalityModel):
    """
    Constant per-week survival rate applied to all age bins.

    Each timestep: ``density *= survival_per_week``.

    Parameters
    ----------
    survival_per_week : float
        Fraction of individuals surviving per week.  Must be in (0, 1].
    """

    def __init__(self, survival_per_week: float) -> None:
        super().__init__()
        if not (0.0 < survival_per_week <= 1.0):
            raise ValueError("survival_per_week must be in (0, 1]")
        self.survival = survival_per_week

    def step(self, density: np.ndarray, timestep: int) -> np.ndarray:
        died = float(density.sum()) * (1.0 - self.survival)
        density *= self.survival
        self._log.append({"timestep": timestep, "total_mortality": died})
        return density
