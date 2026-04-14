"""
Far-field propagule dispersal model for the population simulation.

Adults produce propagules each timestep.  Those propagules drift via ocean
currents and are captured by the pre-computed Lagrangian connectivity tensor,
which maps (source cell, travel-time weeks) → (destination cell, probability).
Propagules are routed through the connectivity tensor and accumulate in a
settling tensor tracking future arrivals; at each timestep the slot
representing "arriving now" is returned to the population model as new
recruits at age-bin 0.

## Unit convention — discrete organisms (``organism_type="discrete"``)

The state variable is an **individual count per cell** (ind/cell).
Fecundity is *propagules per individual per week*, so the raw einsum output
is propagules produced per source cell:

    propagules_produced [propagules/cell] = Σ_a f(a) [prop/ind/wk]
                                              * n(a, y, x) [ind/cell]

Routing through the connectivity tensor requires stochastic integer sampling
(the Poisson distribution is over discrete propagules):

    n_settling = Poisson(propagules_produced * connectivity_weight)

The integer settler count is added directly to the destination cell —
no area conversion is needed because the state is already a per-cell count.

## Unit convention — continuous organisms (``organism_type="continuous"``)

The state variable is **coverage fraction** in [0, 1].
Fecundity is *propagules per unit-coverage per week*:

    propagules_produced [propagules/cell] = Σ_a f(a) [prop/cov/wk]
                                              * n(a, y, x) [coverage]

Settlement is deterministic (no Poisson sampling):

    coverage_added = propagules_produced * connectivity_weight
                     * coverage_per_fragment

where ``coverage_per_fragment`` [coverage/propagule] converts the arriving
propagule count into a coverage increment.  Default: 1e-8 (calibrated for
a 100 * 100 m cell).

``FarFieldDispersal`` is constructed from:

    connectivity : dict
        Loaded by :func:`eradication.connectivity.load.load_connectivity`.
        Keys: ``src_x, src_y, dst_x, dst_y, age, weight`` (all 1-D arrays).
        ``age`` = travel time in weeks; ``weight`` = settlement probability.
    fecundity_by_week : (max_age_weeks,) float32 array
        Propagules released per base-unit (individual or unit-coverage) per
        week for each organism age.  Zero for juvenile ages — no explicit
        masking required.
    competency_period_weeks : int
        Length of the settling tensor age axis.  Must satisfy
        ``connectivity["age"].max() < competency_period_weeks``; raises
        ``ValueError`` otherwise.
    rng : np.random.Generator
        Shared random-number generator (used for Poisson sampling in discrete
        mode).  Pass the generator created by the top-level ``PopulationModel``.
    organism_type : str
        ``"discrete"`` (default) or ``"continuous"``.
    coverage_per_fragment : float
        Coverage fraction introduced by one settling propagule (continuous
        organisms only).  Default: 1e-8.

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
        Name of the rate value key (e.g. ``"propagules_per_week"``).
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
        Column name to read (e.g. ``"propagules_per_week"``).
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
    Far-field propagule dispersal via a pre-computed Lagrangian connectivity tensor.

    Parameters
    ----------
    connectivity : dict
        Keys: ``src_x, src_y, dst_x, dst_y, age, weight`` (all 1-D arrays).
        ``age`` is travel time in weeks (int); ``weight`` is float32 settlement
        probability in [0, 1].
    fecundity_by_week : (n_ages,) array-like
        Propagules released per base-unit (individual or unit-coverage) per
        week for each organism age bin.
    competency_period_weeks : int
        Size of the settling tensor age axis (= max travel time + 1).
        ``connectivity["age"].max()`` must be strictly less than this value.
    ny, nx : int
        Spatial grid dimensions.
    rng : np.random.Generator
        Shared random-number generator used for Poisson sampling (discrete mode).
    organism_type : str
        ``"discrete"`` (default) or ``"continuous"``.
    coverage_per_fragment : float
        Coverage fraction introduced by one settling propagule.  Used only
        when ``organism_type="continuous"``.  Default: 1e-8.
    """

    def __init__(
        self,
        connectivity: dict,
        fecundity_by_week: np.ndarray,
        competency_period_weeks: int,
        ny: int,
        nx: int,
        rng: np.random.Generator,
        organism_type: str = "discrete",
        coverage_per_fragment: float = 1e-8,
    ) -> None:
        max_travel = int(connectivity["age"].max()) if len(connectivity["age"]) > 0 else 0
        if max_travel >= competency_period_weeks:
            raise ValueError(
                f"connectivity['age'].max()={max_travel} must be strictly less than "
                f"competency_period_weeks={competency_period_weeks}. "
                "Ensure the connectivity tensor was built with the same competency period."
            )

        if len(connectivity["dst_x"]) > 0:
            if connectivity["dst_x"].max() >= nx or connectivity["dst_y"].max() >= ny:
                raise ValueError(
                    f"Connectivity dst indices out of bounds for grid ({ny}, {nx}): "
                    f"dst_x max={connectivity['dst_x'].max()}, "
                    f"dst_y max={connectivity['dst_y'].max()}. "
                    "Re-run run_lagrangian.py to regenerate connectivity.npz."
                )

        if organism_type not in ("discrete", "continuous"):
            raise ValueError(
                f"organism_type must be 'discrete' or 'continuous', got {organism_type!r}."
            )

        self._fecundity = np.asarray(fecundity_by_week, dtype=np.float32)
        self._competency = competency_period_weeks
        self._ny = ny
        self._nx = nx
        self._organism_type = organism_type
        self._coverage_per_fragment = float(coverage_per_fragment)

        # Store connectivity arrays (defensive copies, fixed dtypes)
        self._src_y = connectivity["src_y"].astype(np.int32)
        self._src_x = connectivity["src_x"].astype(np.int32)
        self._dst_y = connectivity["dst_y"].astype(np.int32)
        self._dst_x = connectivity["dst_x"].astype(np.int32)
        self._travel_age = connectivity["age"].astype(np.int32)
        self._weight = connectivity["weight"].astype(np.float32)

        # Settling tensor: slot [k] = propagules/coverage arriving in k more weeks
        self._settling = np.zeros((competency_period_weeks, ny, nx), dtype=np.float32)

        self._rng = rng
        self._log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, density: np.ndarray, timestep: int) -> np.ndarray:
        """
        Advance the propagule dispersal model by one timestep.

        Parameters
        ----------
        density : (n_ages, ny, nx) float32 array
            Current age-structured population state.
            Units: individuals/cell (discrete) | coverage [0-1] (continuous).
        timestep : int
            Current model timestep index (0-based).

        Returns
        -------
        settlers : (ny, nx) float32 array
            Propagules/coverage settling this timestep — to be added to age-bin 0.
            Units match the state array: individuals/cell (discrete) |
            coverage fraction (continuous).
        """
        # 1. Collect propagules settling this timestep
        settlers = self._settling[0].copy()

        # 2. Shift settling tensor: slot k → k-1; zero out the now-empty last slot
        self._settling[:-1] = self._settling[1:]
        self._settling[-1] = 0.0

        # 3. Compute propagules produced by the current population.
        #    fecundity is 0 for juvenile age bins — no masking needed.
        #
        #    discrete:   P(y, x) = Σ_a f(a) [prop/ind/wk]  * n(a, y, x) [ind/cell]
        #                        = propagules produced at cell (y, x)
        #    continuous: P(y, x) = Σ_a f(a) [prop/cov/wk]  * n(a, y, x) [coverage]
        #                        = propagules produced at cell (y, x)
        propagules_produced = np.einsum(
            "a,ayx->yx", self._fecundity, density
        ).astype(np.float32)

        # 4. Route propagules through connectivity tensor.
        #
        #    Each connectivity weight w is the per-propagule probability that a
        #    single propagule released at source s settles at destination d after
        #    k weeks (Binomial trial).  Given N propagules at the source, the
        #    number of settlers is Binomial(N, w) ≈ Poisson(N * w).
        #
        #    discrete:   propagules_produced is a per-cell count [prop/cell].
        #                Poisson-sample the integer number of settlers and add
        #                directly to the destination cell count — no area
        #                conversion required.
        #
        #    continuous: propagules_produced is a per-cell count derived from
        #                coverage fractions.  Settlement is deterministic; the
        #                arriving propagule count is scaled by
        #                coverage_per_fragment to give a coverage increment.
        if len(self._src_x) > 0:
            src_propagules = propagules_produced[self._src_y, self._src_x].astype(np.float64)

            if self._organism_type == "discrete":
                # Poisson-sample integer settler counts; add directly as ind/cell
                expected = src_propagules * self._weight.astype(np.float64)
                settling_contribution = self._rng.poisson(expected).astype(np.float32)
            else:  # "continuous"
                # Deterministic; scale by coverage_per_fragment → coverage increment
                settling_contribution = (
                    src_propagules * self._weight.astype(np.float64)
                    * self._coverage_per_fragment
                ).astype(np.float32)

            np.add.at(
                self._settling,
                (self._travel_age, self._dst_y, self._dst_x),
                settling_contribution,
            )

        # 5. Log
        self._log.append({
            "timestep": timestep,
            "total_propagules_produced": float(propagules_produced.sum()),
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
        rng: np.random.Generator,
        organism_type: str = "discrete",
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
        rng : np.random.Generator
            Shared random-number generator passed down from ``PopulationModel``.
        organism_type : str
            ``"discrete"`` (default) or ``"continuous"``.
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
            fecundity = _rates_from_steps(organism_cfg["fecundity"], "propagules_per_week", n_ages)
        else:
            fecundity = _rates_from_csv(organism_cfg["fecundity_csv"], "propagules_per_week", n_ages)

        competency_period_weeks = int(
            full_config["connectivity"]["competency_period_weeks"]
        )

        coverage_per_fragment = float(
            organism_cfg.get("coverage_per_fragment", 1e-8)
        )

        return cls(
            connectivity=connectivity,
            fecundity_by_week=fecundity,
            competency_period_weeks=competency_period_weeks,
            ny=ny,
            nx=nx,
            rng=rng,
            organism_type=organism_type,
            coverage_per_fragment=coverage_per_fragment,
        )
