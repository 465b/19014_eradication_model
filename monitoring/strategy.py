"""
Monitoring strategies for the eradication model.

Each strategy receives the population density field (ny, nx) and the current
model timestep integer, determines which cells are surveyed, applies stochastic
detection, and returns a 2D boolean response field.

Detection and response logic (per surveyed cell):
    detected  = (density > 0) AND Bernoulli(detection_probability)
    response  = detected AND (density > response_threshold)

Three strategies are provided:

    FullGridStrategy        — surveys every suitable habitat cell at a fixed interval
    FractionalStrategy      — surveys a random fraction of suitable cells at a fixed interval
    CustomMaskStrategy      — surveys a user-supplied set of cells at a fixed interval

Multiple strategies can be stacked via MonitoringModel, which combines response
fields with OR logic.  An aggregate log (one entry per timestep) is maintained
by each strategy and by MonitoringModel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class MonitoringStrategy(ABC):
    """Abstract base for a single monitoring strategy."""

    def __init__(
        self,
        survey_interval_steps: int,
        detection_probability: float,
        response_threshold: float,
        seed: int = 0,
    ):
        """
        Parameters
        ----------
        survey_interval_steps : int
            Number of model timesteps between surveys.  A survey fires at
            every timestep where ``timestep % survey_interval_steps == 0``.
        detection_probability : float
            Probability [0, 1] of detecting the organism given it is present
            in a surveyed cell.
        response_threshold : float
            Minimum density (ind/m² or coverage fraction) that must be
            exceeded for a detection to trigger a treatment response.
        seed : int
            RNG seed (always seeded for reproducibility).
        """
        if survey_interval_steps < 1:
            raise ValueError("survey_interval_steps must be >= 1")
        self.survey_interval_steps = survey_interval_steps
        self.detection_probability = detection_probability
        self.response_threshold = response_threshold
        self._rng = np.random.default_rng(seed)
        self._log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, density: np.ndarray, timestep: int) -> np.ndarray:
        """
        Evaluate monitoring for one model timestep.

        Parameters
        ----------
        density : (ny, nx) float array
            Population density (or coverage fraction) from the population model.
        timestep : int
            Current model timestep index (0-based).  The strategy uses this
            to determine whether a survey is scheduled and which cells are
            surveyed.

        Returns
        -------
        response : (ny, nx) bool array
            True where a treatment response is triggered this timestep.
        """
        surveyed = self._get_surveyed_mask(timestep, density.shape)

        detected = (
            surveyed
            & (density > 0)
            & (self._rng.random(density.shape) < self.detection_probability)
        )

        response = detected & (density > self.response_threshold)

        self._log.append({
            "timestep": timestep,
            "n_surveyed": int(surveyed.sum()),
            "n_detected": int(detected.sum()),
            "n_responded": int(response.sum()),
        })
        return response

    @property
    def log(self) -> list[dict[str, Any]]:
        """Per-timestep aggregate log for this strategy."""
        return self._log

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_surveyed_mask(self, timestep: int, shape: tuple[int, int]) -> np.ndarray:
        """
        Return (ny, nx) bool array: True where cells are surveyed at this timestep.

        Return all-False if no survey is scheduled at this timestep.
        """

    def _is_survey_step(self, timestep: int) -> bool:
        return timestep % self.survey_interval_steps == 0


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class FullGridStrategy(MonitoringStrategy):
    """
    Surveys every suitable habitat cell simultaneously at a fixed interval.

    All cells in the habitat mask (or all grid cells if no mask is provided)
    are surveyed every ``survey_interval_steps`` model timesteps.
    """

    def __init__(
        self,
        survey_interval_steps: int,
        detection_probability: float,
        response_threshold: float,
        habitat_mask: np.ndarray | None = None,
        seed: int = 0,
    ):
        """
        Parameters
        ----------
        habitat_mask : (ny, nx) bool array, optional
            If provided, only True cells are surveyed.  If None, all cells are
            surveyed.
        """
        super().__init__(survey_interval_steps, detection_probability, response_threshold, seed)
        self.habitat_mask = habitat_mask.astype(bool) if habitat_mask is not None else None

    def _get_surveyed_mask(self, timestep: int, shape: tuple[int, int]) -> np.ndarray:
        if not self._is_survey_step(timestep):
            return np.zeros(shape, dtype=bool)
        if self.habitat_mask is not None:
            return self.habitat_mask
        return np.ones(shape, dtype=bool)


class FractionalStrategy(MonitoringStrategy):
    """
    Surveys a random fraction of suitable habitat cells at a fixed interval.

    At each scheduled survey timestep, ``ceil(survey_fraction * n_suitable)``
    cells are drawn uniformly at random (without replacement) from the suitable
    habitat cells and surveyed.  The random draw uses the seeded RNG so it is
    reproducible but differs between survey events.
    """

    def __init__(
        self,
        survey_interval_steps: int,
        survey_fraction: float,
        detection_probability: float,
        response_threshold: float,
        habitat_mask: np.ndarray | None = None,
        seed: int = 0,
    ):
        """
        Parameters
        ----------
        survey_fraction : float
            Fraction of suitable cells surveyed per event, e.g. ``0.10`` for
            10 %.  Must be in (0, 1].
        habitat_mask : (ny, nx) bool array, optional
            Pool of suitable cells to sample from.  If None, all grid cells
            are eligible.
        """
        super().__init__(survey_interval_steps, detection_probability, response_threshold, seed)
        if not (0.0 < survey_fraction <= 1.0):
            raise ValueError("survey_fraction must be in (0, 1]")
        self.survey_fraction = survey_fraction
        self.habitat_mask = habitat_mask.astype(bool) if habitat_mask is not None else None

    def _get_surveyed_mask(self, timestep: int, shape: tuple[int, int]) -> np.ndarray:
        if not self._is_survey_step(timestep):
            return np.zeros(shape, dtype=bool)

        # Build pool of candidate cell indices.
        if self.habitat_mask is not None:
            candidate_idx = np.argwhere(self.habitat_mask)  # (n_suitable, 2)
        else:
            ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
            candidate_idx = np.column_stack([ys.ravel(), xs.ravel()])

        n_sample = max(1, int(np.ceil(self.survey_fraction * len(candidate_idx))))
        chosen = self._rng.choice(len(candidate_idx), size=n_sample, replace=False)

        mask = np.zeros(shape, dtype=bool)
        rows, cols = candidate_idx[chosen].T
        mask[rows, cols] = True
        return mask


class CustomMaskStrategy(MonitoringStrategy):
    """
    Surveys a fixed, user-supplied set of cells at a fixed interval.

    The set of monitored cells is provided as a boolean array (loaded from a
    netCDF file by the caller).  Only True cells are ever surveyed.
    """

    def __init__(
        self,
        mask: np.ndarray,
        survey_interval_steps: int,
        detection_probability: float,
        response_threshold: float,
        seed: int = 0,
    ):
        """
        Parameters
        ----------
        mask : (ny, nx) bool array
            True = monitored cell, False = never surveyed.
        """
        super().__init__(survey_interval_steps, detection_probability, response_threshold, seed)
        self.mask = mask.astype(bool)

    def _get_surveyed_mask(self, timestep: int, shape: tuple[int, int]) -> np.ndarray:
        if not self._is_survey_step(timestep):
            return np.zeros(shape, dtype=bool)
        return self.mask


# ---------------------------------------------------------------------------
# Stacked model
# ---------------------------------------------------------------------------

class MonitoringModel:
    """
    Stacks multiple monitoring strategies and returns their combined response.

    All strategies are stepped in order; their response fields are combined
    with OR logic (a cell is True if *any* strategy fires there).

    A combined aggregate log is maintained alongside the per-strategy logs.

    Example
    -------
    >>> model = MonitoringModel([
    ...     FullGridStrategy(survey_interval_steps=4,
    ...                      detection_probability=0.7,
    ...                      response_threshold=0.5,
    ...                      habitat_mask=habitat,
    ...                      seed=42),
    ...     FractionalStrategy(survey_interval_steps=2,
    ...                        survey_fraction=0.1,
    ...                        detection_probability=0.9,
    ...                        response_threshold=0.1,
    ...                        habitat_mask=habitat,
    ...                        seed=43),
    ... ])
    >>> response = model.step(density, timestep=8)  # (ny, nx) bool
    """

    def __init__(self, strategies: list[MonitoringStrategy]):
        if not strategies:
            raise ValueError("At least one MonitoringStrategy is required.")
        self.strategies = strategies
        self._log: list[dict[str, Any]] = []

    def step(self, density: np.ndarray, timestep: int) -> np.ndarray:
        """
        Advance all strategies one timestep and return the combined response.

        Parameters
        ----------
        density : (ny, nx) float array
            Population density field from the population model.
        timestep : int
            Current model timestep index (0-based).

        Returns
        -------
        response : (ny, nx) bool array
            OR combination of all strategy response fields.
        """
        combined = np.zeros(density.shape, dtype=bool)
        n_surveyed = 0
        n_detected = 0

        for strategy in self.strategies:
            resp = strategy.step(density, timestep)
            combined |= resp
            last = strategy.log[-1]
            n_surveyed += last["n_surveyed"]
            n_detected += last["n_detected"]

        self._log.append({
            "timestep": timestep,
            # n_surveyed / n_detected summed across strategies; may double-count
            # cells covered by more than one strategy.
            "n_surveyed": n_surveyed,
            "n_detected": n_detected,
            # n_responded is the union: unique cells with a True response.
            "n_responded": int(combined.sum()),
        })
        return combined

    @property
    def log(self) -> list[dict[str, Any]]:
        """
        Combined aggregate log: one entry per timestep.

        Keys: ``timestep``, ``n_surveyed``, ``n_detected``, ``n_responded``.
        ``n_surveyed`` and ``n_detected`` are summed across strategies and may
        exceed the grid size when strategies overlap spatially.
        ``n_responded`` is the count of unique cells with a True response.
        """
        return self._log

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        dt_weeks: int,
        seed: int = 0,
        habitat_mask: np.ndarray | None = None,
        custom_masks: dict[str, np.ndarray] | None = None,
    ) -> "MonitoringModel":
        """
        Build a MonitoringModel from the ``monitoring`` config section.

        The config must contain a ``strategies`` list; each entry describes one
        strategy and is built independently.  Strategies are seeded with
        ``seed + i`` (where ``i`` is the list index) so they use independent
        but deterministic RNGs.

        Parameters
        ----------
        cfg : dict
            The ``monitoring`` section of the scenario YAML.  Must contain a
            ``strategies`` key with a list of strategy config dicts.
        dt_weeks : int
            Model timestep size in weeks (from ``temporal.dt_weeks``).
        seed : int
            Base RNG seed.  Strategy ``i`` receives seed ``seed + i``.
        habitat_mask : (ny, nx) bool array, optional
            Suitable-habitat mask used by ``full_grid`` and ``random_fraction``
            strategies.
        custom_masks : dict[str, ndarray], optional
            Pre-loaded boolean grids for ``custom_mask`` strategies, keyed by
            the ``mask_file`` path string from the config.  The caller is
            responsible for loading these from the netCDF files listed in the
            config before calling ``from_config``.
        """
        custom_masks = custom_masks or {}
        strategies: list[MonitoringStrategy] = []

        for i, entry in enumerate(cfg["strategies"]):
            strategies.append(
                cls._build_one(
                    entry=entry,
                    dt_weeks=dt_weeks,
                    seed=seed + i,
                    habitat_mask=habitat_mask,
                    custom_masks=custom_masks,
                )
            )

        return cls(strategies)

    @staticmethod
    def _build_one(
        entry: dict[str, Any],
        dt_weeks: int,
        seed: int,
        habitat_mask: np.ndarray | None,
        custom_masks: dict[str, np.ndarray],
    ) -> MonitoringStrategy:
        """Instantiate a single strategy from one entry in the ``strategies`` list."""
        strategy_type = entry["type"]
        det_prob = float(entry["detection_probability"])
        resp_thresh = float(entry.get("response_threshold", 0.0))

        interval_weeks = entry["survey_interval_weeks"]
        if interval_weeks % dt_weeks != 0:
            raise ValueError(
                f"survey_interval_weeks={interval_weeks} is not an integer multiple "
                f"of dt_weeks={dt_weeks}"
            )
        interval_steps = interval_weeks // dt_weeks

        if strategy_type == "full_grid":
            return FullGridStrategy(
                survey_interval_steps=interval_steps,
                detection_probability=det_prob,
                response_threshold=resp_thresh,
                habitat_mask=habitat_mask,
                seed=seed,
            )

        if strategy_type == "random_fraction":
            return FractionalStrategy(
                survey_interval_steps=interval_steps,
                survey_fraction=float(entry["survey_fraction"]),
                detection_probability=det_prob,
                response_threshold=resp_thresh,
                habitat_mask=habitat_mask,
                seed=seed,
            )

        if strategy_type == "custom_mask":
            mask_file = entry["mask_file"]
            if mask_file not in custom_masks:
                raise ValueError(
                    f"custom_mask strategy requires a pre-loaded array for "
                    f"mask_file={mask_file!r} — pass it via custom_masks dict"
                )
            return CustomMaskStrategy(
                mask=custom_masks[mask_file],
                survey_interval_steps=interval_steps,
                detection_probability=det_prob,
                response_threshold=resp_thresh,
                seed=seed,
            )

        raise ValueError(f"Unknown monitoring strategy type: {strategy_type!r}")
