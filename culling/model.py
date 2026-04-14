"""
Eradication (culling) model.

Consumes the boolean response field produced by MonitoringModel and returns a
``cull_efficiency`` array that the population model applies as a mortality term.

Two culling implementations are provided:

    FlatFractionCulling     — deterministically removes a fixed fraction of the
                              population in every treated cell.
    FlatProbabilityCulling  — stochastically removes the entire population with
                              probability ``p_full_removal``; falls back to a
                              smaller ``partial_fraction`` otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EradicationModel(ABC):
    """Abstract base for a culling implementation."""

    def __init__(self) -> None:
        self._log: list[dict[str, Any]] = []

    @abstractmethod
    def step(self, response: np.ndarray, timestep: int) -> np.ndarray:
        """
        Compute the cull efficiency field for one model timestep.

        Parameters
        ----------
        response : (ny, nx) bool array
            Treatment response field from ``MonitoringModel.step()``.
            True = treat this cell.
        timestep : int
            Current model timestep index (0-based).

        Returns
        -------
        cull_efficiency : (ny, nx) float32 array
            Fraction of density to remove per cell; unit-agnostic —
            applies to individuals/cell (discrete) or coverage (continuous)
            equally.  0.0 = untreated, 1.0 = complete removal.
        """

    @property
    def log(self) -> list[dict[str, Any]]:
        """Aggregate log: one entry per timestep."""
        return self._log

    @classmethod
    def from_config(cls, cfg: dict[str, Any], seed: int = 0) -> "EradicationModel":
        """
        Build an EradicationModel from the ``eradication`` config section.

        Dispatches on ``cfg["method"]``:
          ``"flat_fraction"``    → FlatFractionCulling
          ``"flat_probability"`` → FlatProbabilityCulling

        Parameters
        ----------
        cfg : dict
            The ``eradication`` section of the scenario YAML.
        seed : int
            RNG seed forwarded to FlatProbabilityCulling (ignored for
            FlatFractionCulling which is deterministic).
        """
        method = cfg["method"]

        if method == "flat_fraction":
            return FlatFractionCulling(
                cull_fraction=float(cfg["cull_fraction"]),
            )

        if method == "flat_probability":
            return FlatProbabilityCulling(
                p_full_removal=float(cfg["p_full_removal"]),
                partial_fraction=float(cfg["partial_fraction"]),
                seed=seed,
            )

        raise ValueError(
            f"Unknown eradication method {method!r}. "
            "Must be 'flat_fraction' or 'flat_probability'."
        )


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class FlatFractionCulling(EradicationModel):
    """
    Deterministic culling: every treated cell loses exactly ``cull_fraction``
    of its population.

    Parameters
    ----------
    cull_fraction : float
        Fraction of density removed in treated cells; applies to
        individuals/cell (discrete) or coverage fraction (continuous) equally.
        Must be in (0, 1].

    Example
    -------
    >>> model = FlatFractionCulling(cull_fraction=0.9)
    >>> cull = model.step(response, timestep=4)  # 0.0 or 0.9 per cell
    """

    def __init__(self, cull_fraction: float) -> None:
        super().__init__()
        if not (0.0 < cull_fraction <= 1.0):
            raise ValueError("cull_fraction must be in (0, 1].")
        self.cull_fraction = cull_fraction

    def step(self, response: np.ndarray, timestep: int) -> np.ndarray:
        cull = np.where(response, self.cull_fraction, 0.0).astype(np.float32)
        self._log.append({
            "timestep": timestep,
            "n_treated": int(response.sum()),
        })
        return cull


class FlatProbabilityCulling(EradicationModel):
    """
    Stochastic culling: each treated cell independently draws whether the
    treatment achieves full removal or only partial removal.

    For each treated cell:
      - with probability ``p_full_removal`` → cull_efficiency = 1.0
      - otherwise                           → cull_efficiency = ``partial_fraction``

    Parameters
    ----------
    p_full_removal : float
        Per-cell probability of complete removal.  Must be in (0, 1].
    partial_fraction : float
        Fraction removed when full removal does not occur.  Must be in (0, 1).
    seed : int
        RNG seed (always seeded for reproducibility).

    Example
    -------
    >>> model = FlatProbabilityCulling(p_full_removal=0.3, partial_fraction=0.9, seed=0)
    >>> cull = model.step(response, timestep=4)  # 0.0, 0.9, or 1.0 per cell
    """

    def __init__(
        self,
        p_full_removal: float,
        partial_fraction: float,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not (0.0 < p_full_removal <= 1.0):
            raise ValueError("p_full_removal must be in (0, 1].")
        if not (0.0 < partial_fraction < 1.0):
            raise ValueError("partial_fraction must be in (0, 1).")
        self.p_full_removal = p_full_removal
        self.partial_fraction = partial_fraction
        self._rng = np.random.default_rng(seed)

    def step(self, response: np.ndarray, timestep: int) -> np.ndarray:
        full = response & (self._rng.random(response.shape) < self.p_full_removal)
        partial = response & ~full

        cull = np.zeros(response.shape, dtype=np.float32)
        cull[full] = 1.0
        cull[partial] = self.partial_fraction

        self._log.append({
            "timestep": timestep,
            "n_treated": int(response.sum()),
            "n_full_removal": int(full.sum()),
            "n_partial": int(partial.sum()),
        })
        return cull
