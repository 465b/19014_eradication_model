"""
Natural mortality models for the population simulation.

Natural mortality is applied *independently* of eradication-induced
mortality (which is handled by the culling module).  Each model receives
the full age-structured density and returns the updated density after
applying age-dependent die-off.

One implementation is provided:

    AgeDependentSurvival — per-week survival rate array applied to each
                           age bin independently each timestep.

## Organism-type behaviour

``survival_per_week`` is dimensionless [0-1] for both organism types, but
the operation applied differs:

* **discrete** — ``Binomial(N_individuals, survival_per_week)`` per age bin per cell.
  Each individual survives independently; populations can reach exactly zero.
* **continuous** — multiplicative decay: ``coverage * survival_per_week``.

Configuration supports two mutually exclusive modes via the ``organism``
config section:

    survival:              piecewise-constant step-function (inline YAML)
    survival_csv: <path>   CSV file with one row per organism-age week

Omitting both keys defaults to no natural mortality (survival = 1.0 for
all age bins).
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate-array helpers (private)
# ---------------------------------------------------------------------------

def _rates_from_steps(
    steps: list[dict],
    rate_key: str,
    n_weeks: int,
) -> np.ndarray:
    """Build a float32 per-week array from a piecewise-constant step spec.

    Parameters
    ----------
    steps : list of dicts with keys ``above_week`` and *rate_key*
    rate_key : str
        Name of the rate value key (e.g. ``"survival_per_week"``).
    n_weeks : int
        Length of the output array (= ``max_age_weeks``).

    Validation
    ----------
    - Must be non-empty.
    - First entry must have ``above_week == 0``.
    - ``above_week`` values must be strictly increasing (raises ``ValueError``).
    - Breakpoints >= ``n_weeks`` are ignored with a warning.
    """
    if not steps:
        raise ValueError("Step-function spec must not be empty.")

    sorted_steps = sorted(steps, key=lambda s: int(s["above_week"]))

    if int(sorted_steps[0]["above_week"]) != 0:
        raise ValueError(
            "Step-function spec must start with above_week: 0, "
            f"got {sorted_steps[0]['above_week']}."
        )

    breakpoints = [int(s["above_week"]) for s in sorted_steps]
    for i in range(1, len(breakpoints)):
        if breakpoints[i] <= breakpoints[i - 1]:
            raise ValueError(
                f"above_week values must be strictly increasing; "
                f"got {breakpoints[i - 1]} then {breakpoints[i]}."
            )

    for bp in breakpoints:
        if bp >= n_weeks:
            warnings.warn(
                f"above_week={bp} is >= max_age_weeks={n_weeks}; "
                "this breakpoint applies to ages beyond the model range and will be ignored.",
                stacklevel=3,
            )

    out = np.empty(n_weeks, dtype=np.float32)
    current_value = float(sorted_steps[0][rate_key])
    step_idx = 1
    for w in range(n_weeks):
        while step_idx < len(sorted_steps) and int(sorted_steps[step_idx]["above_week"]) <= w:
            current_value = float(sorted_steps[step_idx][rate_key])
            step_idx += 1
        out[w] = current_value
    return out


def _rates_from_csv(
    path: str | Path,
    column: str,
    n_weeks: int,
) -> np.ndarray:
    """Read a per-week rate array from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.  Must have a header row containing *column*.
    column : str
        Column name to read (e.g. ``"survival_per_week"``).
    n_weeks : int
        Expected number of rows (= ``max_age_weeks``).

    Raises
    ------
    ValueError
        If the file has fewer than *n_weeks* data rows.
    """
    data = np.genfromtxt(path, delimiter=",", names=True)
    values = data[column]
    if len(values) < n_weeks:
        raise ValueError(
            f"CSV file {path!r} has {len(values)} rows but "
            f"max_age_weeks={n_weeks} rows are required."
        )
    return values[:n_weeks].astype(np.float32)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


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
    def from_config(
        cls,
        organism_cfg: dict,
        organism_type: str = "discrete",
        rng: np.random.Generator | None = None,
    ) -> "MortalityModel":
        """
        Factory: build a mortality model from the ``organism`` config section.

        Reads ``max_age_weeks`` to set the length of the survival array.
        Dispatches on the presence of ``survival`` or ``survival_csv`` keys:

            ``survival``     (list of step dicts) → inline step-function
            ``survival_csv`` (file path)           → CSV file
            neither present                        → no natural mortality (survival = 1.0)

        Parameters
        ----------
        organism_cfg : dict
            ``config["organism"]`` section.
        organism_type : str
            ``"discrete"`` or ``"continuous"`` — controls whether Binomial
            sampling or multiplicative decay is used in ``step()``.
        rng : np.random.Generator, optional
            Shared RNG for Binomial draws (discrete mode only).  If ``None``
            a fresh default generator is used; pass the model-level generator
            for reproducible runs.
        """
        n_ages = int(organism_cfg["max_age_weeks"])

        has_inline = "survival" in organism_cfg
        has_csv = "survival_csv" in organism_cfg

        if has_inline and has_csv:
            raise ValueError(
                "Specify exactly one of 'survival' or 'survival_csv' in the organism config, not both."
            )

        if has_inline:
            survival = _rates_from_steps(organism_cfg["survival"], "survival_per_week", n_ages)
        elif has_csv:
            survival = _rates_from_csv(organism_cfg["survival_csv"], "survival_per_week", n_ages)
        else:
            log.warning(
                "Natural mortality disabled: neither 'survival' nor 'survival_csv' found in organism config."
            )
            survival = np.ones(n_ages, dtype=np.float32)

        return AgeDependentSurvival(survival, organism_type=organism_type, rng=rng)


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class AgeDependentSurvival(MortalityModel):
    """
    Per-week survival rate applied independently to each age bin.

    The operation depends on ``organism_type``:

    * **discrete** — ``Binomial(N, survival[a])`` per cell per age bin.
      Requires integer-valued density; each individual survives independently.
      Populations can reach exactly zero (no ghost densities).
    * **continuous** — ``density[a] *= survival[a]`` for all age bins *a*.
      Multiplicative decay of coverage fraction.

    Parameters
    ----------
    survival_by_week : (n_ages,) float32 array
        Per-week survival probability (discrete) or survival fraction
        (continuous) for each age bin.  All values must be in (0, 1].
    organism_type : str
        ``"discrete"`` or ``"continuous"``.
    rng : np.random.Generator, optional
        Shared RNG for Binomial draws (discrete mode only).
    """

    def __init__(
        self,
        survival_by_week: np.ndarray,
        organism_type: str = "discrete",
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        survival_by_week = np.asarray(survival_by_week, dtype=np.float32)
        if not np.all((survival_by_week > 0.0) & (survival_by_week <= 1.0)):
            raise ValueError("All survival_per_week values must be in (0, 1].")
        if organism_type not in ("discrete", "continuous"):
            raise ValueError(
                f"organism_type must be 'discrete' or 'continuous', got {organism_type!r}."
            )
        self._survival = survival_by_week
        self._organism_type = organism_type
        self._rng = rng if rng is not None else np.random.default_rng()

    def step(self, density: np.ndarray, timestep: int) -> np.ndarray:
        total_before = float(density.sum())

        if self._organism_type == "discrete":
            # Binomial survival: each individual independently survives with
            # probability survival[a].  Operates on integer counts per cell.
            for a, p in enumerate(self._survival):
                n = density[a].astype(np.int64)
                density[a] = self._rng.binomial(n, float(p)).astype(np.float32)
        else:
            # Multiplicative decay of coverage fraction
            density *= self._survival[:, np.newaxis, np.newaxis]

        total_after = float(density.sum())
        self._log.append({
            "timestep": timestep,
            "total_mortality": total_before - total_after,
        })
        return density
