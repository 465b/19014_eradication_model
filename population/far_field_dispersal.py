"""
Far-field larval dispersal model for the population simulation.

Adults produce larvae each timestep.  Those larvae drift via ocean currents
and are captured by the pre-computed Lagrangian connectivity tensor, which
maps (source cell, travel-time weeks) → (destination cell, probability).
Larvae are routed through the connectivity tensor and accumulate in a
settling tensor tracking future arrivals; at each timestep the slot
representing "arriving now" is returned to the population model as new
recruits at age-bin 0.

## Unit convention

The state variable throughout the model is **areal density** (ind/m²).
Fecundity is *larvae per individual per week*, so the raw output of the
einsum is also a density:

    larvae_produced [larvae/m²] = Σ_a fecundity[a] [larvae/ind/wk]
                                     × density[a] [ind/m²]

Routing through the connectivity tensor requires a **count** (the Poisson
distribution is over discrete larvae), so for ``organism_type="discrete"``
we convert:

    count = larvae_produced × cell_area_m2   → Poisson sample → density = count / cell_area_m2

For ``organism_type="continuous"`` (future: coverage fraction) the
propagation is deterministic and no area conversion is needed; larvae
(or rather propagules) are simply multiplied by the connectivity weight
and accumulated in the settling tensor as-is.

``FarFieldDispersal`` is constructed from:

    connectivity : dict
        Loaded by :func:`eradication.connectivity.load.load_connectivity`.
        Keys: ``src_x, src_y, dst_x, dst_y, age, weight`` (all 1-D arrays).
        ``age`` = travel time in weeks; ``weight`` = settlement probability.
    fecundity_by_week : (max_age_weeks,) float32 array
        Larvae released per individual per week for each organism age.
        Zero for juvenile ages — no explicit masking required.
    competency_period_weeks : int
        Length of the settling tensor age axis.  Must satisfy
        ``connectivity["age"].max() < competency_period_weeks``; raises
        ``ValueError`` otherwise.
    cell_area_m2 : float
        Area of one grid cell in m².  Used for the density↔count conversion
        in discrete-organism mode.  Derived from ``spatial.resolution_m²``.
    organism_type : str
        ``"discrete"`` (default) or ``"continuous"``.

Configuration (``organism`` config section) supports two mutually exclusive
fecundity specs:

    fecundity:              piecewise-constant step-function (inline YAML)
    fecundity_csv: <path>   CSV with one row per organism-age week
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate-array helpers (private — duplicated from mortality.py by design)
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
        Name of the rate value key (e.g. ``"larvae_per_week"``).
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
        Column name to read (e.g. ``"larvae_per_week"``).
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
# FarFieldDispersal
# ---------------------------------------------------------------------------


class FarFieldDispersal:
    """
    Far-field larval dispersal via a pre-computed Lagrangian connectivity tensor.

    Parameters
    ----------
    connectivity : dict
        Keys: ``src_x, src_y, dst_x, dst_y, age, weight`` (all 1-D arrays).
        ``age`` is travel time in weeks (int); ``weight`` is float32 settlement
        probability in [0, 1].
    fecundity_by_week : (n_ages,) array-like
        Larvae released per individual per week for each organism age bin.
    competency_period_weeks : int
        Size of the settling tensor age axis (= max travel time + 1).
        ``connectivity["age"].max()`` must be strictly less than this value.
    ny, nx : int
        Spatial grid dimensions.
    rng_seed : int
        Seed for the Poisson random number generator.
    """

    def __init__(
        self,
        connectivity: dict,
        fecundity_by_week: np.ndarray,
        competency_period_weeks: int,
        ny: int,
        nx: int,
        cell_area_m2: float = 1.0,
        organism_type: str = "discrete",
        rng_seed: int = 0,
    ) -> None:
        max_travel = int(connectivity["age"].max()) if len(connectivity["age"]) > 0 else 0
        if max_travel >= competency_period_weeks:
            raise ValueError(
                f"connectivity['age'].max()={max_travel} must be strictly less than "
                f"competency_period_weeks={competency_period_weeks}. "
                "Ensure the connectivity tensor was built with the same competency period."
            )

        if organism_type not in ("discrete", "continuous"):
            raise ValueError(
                f"organism_type must be 'discrete' or 'continuous', got {organism_type!r}."
            )

        self._fecundity = np.asarray(fecundity_by_week, dtype=np.float32)
        self._competency = competency_period_weeks
        self._ny = ny
        self._nx = nx
        self._cell_area_m2 = float(cell_area_m2)
        self._organism_type = organism_type

        # Store connectivity arrays (defensive copies, fixed dtypes)
        self._src_y = connectivity["src_y"].astype(np.int32)
        self._src_x = connectivity["src_x"].astype(np.int32)
        self._dst_y = connectivity["dst_y"].astype(np.int32)
        self._dst_x = connectivity["dst_x"].astype(np.int32)
        self._travel_age = connectivity["age"].astype(np.int32)
        self._weight = connectivity["weight"].astype(np.float32)

        # Settling tensor: slot [k] = larvae arriving in k more weeks
        self._settling = np.zeros((competency_period_weeks, ny, nx), dtype=np.float32)

        self._rng = np.random.default_rng(rng_seed)
        self._log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, density: np.ndarray, timestep: int) -> np.ndarray:
        """
        Advance the larval dispersal model by one timestep.

        Parameters
        ----------
        density : (n_ages, ny, nx) float32 array
            Current age-structured population density.
        timestep : int
            Current model timestep index (0-based).

        Returns
        -------
        settlers : (ny, nx) float32 array
            Larvae that settle this timestep — to be added to age-bin 0.
        """
        # 1. Collect larvae settling this timestep
        settlers = self._settling[0].copy()

        # 2. Shift settling tensor: slot k → k-1; zero out the now-empty last slot
        self._settling[:-1] = self._settling[1:]
        self._settling[-1] = 0.0

        # 3. Compute larvae produced by the current population.
        #    fecundity is 0 for juvenile age bins — no masking needed.
        larvae_produced = np.einsum(
            "a,ayx->yx", self._fecundity, density
        ).astype(np.float32)

        # 4. Route larvae through connectivity tensor.
        #
        #    Each connectivity weight w is the per-larva probability that a
        #    single larva released at source s settles at destination d after
        #    k weeks (Binomial trial).  Given N larvae at the source, the
        #    number of settlers is Binomial(N, w) ≈ Poisson(N × w) — the
        #    Poisson approximation holds when N is large and w is small.
        #
        #    "discrete" organisms: larvae_produced is a density [larvae/m²].
        #    We convert to a count (N = density × cell_area_m2) before
        #    sampling, then convert the settler count back to density.
        #
        #    "continuous" organisms: propagules are a dimensionless fraction;
        #    no area conversion is needed and settlement is deterministic
        #    (the Binomial / Poisson interpretation does not apply).
        if len(self._src_x) > 0:
            src_larvae = larvae_produced[self._src_y, self._src_x].astype(np.float64)

            if self._organism_type == "discrete":
                # density [larvae/m²] → count [larvae]
                expected = src_larvae * self._cell_area_m2 * self._weight.astype(np.float64)
                n_settling = self._rng.poisson(expected)
                # count [settlers] → density [settlers/m²]
                settling_contribution = n_settling / self._cell_area_m2
            else:  # "continuous"
                settling_contribution = src_larvae * self._weight.astype(np.float64)

            np.add.at(
                self._settling,
                (self._travel_age, self._dst_y, self._dst_x),
                settling_contribution,
            )

        # 5. Log
        self._log.append({
            "timestep": timestep,
            "total_larvae_produced": float(larvae_produced.sum()),
            "total_settling": float(settlers.sum()),
        })

        return settlers

    @property
    def log(self) -> list[dict[str, Any]]:
        return self._log

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        organism_cfg: dict,
        connectivity: dict,
        full_config: dict,
        ny: int,
        nx: int,
        cell_area_m2: float = 1.0,
        organism_type: str = "discrete",
        rng_seed: int = 0,
    ) -> "FarFieldDispersal":
        """
        Build a FarFieldDispersal from config dicts.

        Parameters
        ----------
        organism_cfg : dict
            ``config["organism"]`` section.  Must contain ``max_age_weeks``
            and exactly one of ``fecundity`` or ``fecundity_csv``.
        connectivity : dict
            Loaded connectivity tensor (see :mod:`eradication.connectivity.load`).
        full_config : dict
            Full scenario config — used to read
            ``full_config["connectivity"]["competency_period_weeks"]``.
        ny, nx : int
            Spatial grid dimensions.
        cell_area_m2 : float
            Area of one grid cell in m² — used for the density↔count
            conversion in discrete-organism mode.  Pass
            ``spatial.resolution_m ** 2`` from the full config.
        organism_type : str
            ``"discrete"`` (default) or ``"continuous"``.
        rng_seed : int
            Seed for the Poisson RNG.
        """
        n_ages = int(organism_cfg["max_age_weeks"])

        has_inline = "fecundity" in organism_cfg
        has_csv = "fecundity_csv" in organism_cfg

        if has_inline and has_csv:
            raise ValueError(
                "Specify exactly one of 'fecundity' or 'fecundity_csv' in the organism config, not both."
            )
        if not has_inline and not has_csv:
            raise ValueError(
                "organism config must contain 'fecundity' or 'fecundity_csv'."
            )

        if has_inline:
            fecundity = _rates_from_steps(organism_cfg["fecundity"], "larvae_per_week", n_ages)
        else:
            fecundity = _rates_from_csv(organism_cfg["fecundity_csv"], "larvae_per_week", n_ages)

        competency_period_weeks = int(
            full_config["connectivity"]["competency_period_weeks"]
        )

        return cls(
            connectivity=connectivity,
            fecundity_by_week=fecundity,
            competency_period_weeks=competency_period_weeks,
            ny=ny,
            nx=nx,
            cell_area_m2=cell_area_m2,
            organism_type=organism_type,
            rng_seed=rng_seed,
        )
