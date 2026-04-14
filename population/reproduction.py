"""
Growth / recruitment models for the population simulation.

Each model receives the current total density field and returns a
``(ny, nx)`` array of *new recruits* to be placed into the youngest
age bin.  The growth model does **not** modify the existing density
in-place — the caller (:class:`PopulationModel`) is responsible for
feeding recruits into the age structure.

Growth models compute a raw recruitment flux only.  Density-dependent
limitation via a carrying capacity *K* is applied externally by the
caller using a configurable suppression factor (see
``carrying_capacity_suppression`` in the schema) — this keeps K logic
in one place and applies it uniformly to growth, far-field dispersal,
and any other additive process.

## Organism-type behaviour

``growth_rate_per_week`` [1/week] is dimensionless for both organism types,
but the operation differs:

* **discrete** — ``Poisson(r * N_individuals)`` per cell.  Each individual
  independently spawns offspring; recruit count is a stochastic integer.
* **continuous** — ``r * coverage``; deterministic multiplicative flux on
  the coverage fraction.

One concrete implementation is provided:

    ConstantGrowth — ``recruits = r · N``; unbounded per-capita rate.
                     Combined with external K suppression this gives
                     the classic logistic curve.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class GrowthModel(ABC):
    """
    Abstract base for population growth / recruitment models.

    Subclasses compute a raw recruitment flux from the current density
    field.  Any density-dependent ceiling (carrying capacity) is
    applied by :class:`~eradication.population.model.PopulationModel`
    after the raw flux is returned, so subclasses do **not** need to
    be aware of *K*.
    """

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
        Compute the raw recruitment field for one model timestep.

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
            *Not* yet limited by carrying capacity — the caller applies
            the ``max(0, 1 - N/K)`` suppression factor externally.
        """

    @property
    def log(self) -> list[dict[str, Any]]:
        return self._log

    @classmethod
    def from_config(
        cls,
        organism_cfg: dict,
        organism_type: str = "discrete",
        rng: np.random.Generator | None = None,
    ) -> "GrowthModel":
        """
        Factory: dispatch on ``organism_cfg["growth_model"]``.

        Supported values:
            ``"constant"``     → :class:`ConstantGrowth` (default)
            ``"none"``         → :class:`NoGrowth`

        Legacy values ``"logistic"`` and ``"exponential"`` are accepted
        with a deprecation warning and mapped to :class:`ConstantGrowth`
        (carrying-capacity suppression is now applied externally by
        :class:`~eradication.population.model.PopulationModel`).

        If ``growth_rate_per_week`` is absent the growth model is
        disabled (:class:`NoGrowth` returned).

        Parameters
        ----------
        organism_cfg : dict
            ``config["organism"]`` section.
        organism_type : str
            ``"discrete"`` or ``"continuous"`` — controls whether Poisson
            sampling or multiplicative flux is used in ``step()``.
        rng : np.random.Generator, optional
            Shared RNG for Poisson draws (discrete mode only).
        """
        if "growth_rate_per_week" not in organism_cfg:
            log.warning(
                "Growth model disabled: 'growth_rate_per_week' not found in organism config."
            )
            return NoGrowth()

        kind = organism_cfg.get("growth_model", "constant")

        if kind == "none":
            log.warning("Growth model disabled: growth_model set to 'none'.")
            return NoGrowth()

        if kind in ("logistic", "exponential"):
            warnings.warn(
                f"growth_model={kind!r} is deprecated. Use 'constant' instead. "
                "Carrying-capacity suppression is now applied externally by "
                "PopulationModel using the shared max(0, 1-N/K) factor.",
                DeprecationWarning,
                stacklevel=2,
            )
            kind = "constant"

        if kind == "constant":
            return ConstantGrowth(
                growth_rate=float(organism_cfg["growth_rate_per_week"]),
                organism_type=organism_type,
                rng=rng,
            )

        raise ValueError(
            f"Unknown growth_model {kind!r}. Must be 'constant' or 'none'."
        )


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class NoGrowth(GrowthModel):
    """
    No-op growth model — zero recruits every timestep.

    Selected automatically when ``growth_rate_per_week`` is absent from
    the organism config, or explicitly via ``growth_model: "none"``.
    """

    def step(
        self,
        total_density: np.ndarray,
        habitat_mask: np.ndarray,  # noqa: ARG002
        timestep: int,
    ) -> np.ndarray:
        self._log.append({"timestep": timestep, "total_recruits": 0.0})
        return np.zeros_like(total_density)


class ConstantGrowth(GrowthModel):
    """
    Constant per-capita recruitment.

    The operation depends on ``organism_type``:

    * **discrete** — ``Poisson(r * N_individuals)`` per cell.
      Recruits are stochastic integer counts.
    * **continuous** — ``r * coverage``; deterministic flux proportional
      to coverage fraction.

    Density-dependent limitation (carrying capacity *K*) is applied
    externally by the caller via the shared suppression factor, which
    yields the classic logistic curve when K is provided:

    .. math::

        \\text{recruits\\_accepted} = r \\cdot N \\cdot
            \\max\\!\\left(0,\\, 1 - \\frac{N}{K}\\right)

    Parameters
    ----------
    growth_rate : float
        Intrinsic per-week growth rate *r* [1/week].
    organism_type : str
        ``"discrete"`` or ``"continuous"``.
    rng : np.random.Generator, optional
        Shared RNG for Poisson draws (discrete mode only).
    """

    def __init__(
        self,
        growth_rate: float,
        organism_type: str = "discrete",
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        if organism_type not in ("discrete", "continuous"):
            raise ValueError(
                f"organism_type must be 'discrete' or 'continuous', got {organism_type!r}."
            )
        self.r = growth_rate
        self._organism_type = organism_type
        self._rng = rng if rng is not None else np.random.default_rng()

    def step(
        self,
        total_density: np.ndarray,
        habitat_mask: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        if self._organism_type == "discrete":
            # Poisson: each individual independently spawns recruits with rate r
            lam = self.r * total_density.astype(np.float64)
            np.clip(lam, 0.0, None, out=lam)
            recruits = self._rng.poisson(lam).astype(np.float32)
        else:
            # Deterministic multiplicative flux on coverage fraction
            recruits = (self.r * total_density).astype(np.float32)
            np.clip(recruits, 0.0, None, out=recruits)

        recruits[~habitat_mask] = 0.0

        self._log.append({
            "timestep": timestep,
            "total_recruits": float(recruits.sum()),
        })
        return recruits
