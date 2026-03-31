"""
Natural mortality models for the population simulation.

Natural mortality is applied *independently* of eradication-induced
mortality (which is handled by the culling module).  Each model receives
the full age-structured density and returns the updated density after
applying age-dependent die-off.

One implementation is provided:

    AgeDependentSurvival — per-week survival rate array applied to each
                           age bin independently each timestep.

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
    def from_config(cls, organism_cfg: dict) -> "MortalityModel":
        """
        Factory: build a mortality model from the ``organism`` config section.

        Reads ``max_age_weeks`` to set the length of the survival array.
        Dispatches on the presence of ``survival`` or ``survival_csv`` keys:

            ``survival``     (list of step dicts) → inline step-function
            ``survival_csv`` (file path)           → CSV file
            neither present                        → no natural mortality (survival = 1.0)
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

        return AgeDependentSurvival(survival)


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class AgeDependentSurvival(MortalityModel):
    """
    Per-week survival rate applied independently to each age bin.

    Each timestep: ``density[a] *= survival[a]`` for all age bins *a*.

    Parameters
    ----------
    survival_by_week : (n_ages,) float32 array
        Fraction of individuals surviving per week for each age bin.
        All values must be in (0, 1].
    """

    def __init__(self, survival_by_week: np.ndarray) -> None:
        super().__init__()
        survival_by_week = np.asarray(survival_by_week, dtype=np.float32)
        if not np.all((survival_by_week > 0.0) & (survival_by_week <= 1.0)):
            raise ValueError("All survival_per_week values must be in (0, 1].")
        self._survival = survival_by_week

    def step(self, density: np.ndarray, timestep: int) -> np.ndarray:
        total_before = float(density.sum())
        density *= self._survival[:, np.newaxis, np.newaxis]
        total_after = float(density.sum())
        self._log.append({
            "timestep": timestep,
            "total_mortality": total_before - total_after,
        })
        return density
