"""
Population model — main simulation driver.

Ties together the age structure, growth, dispersal, natural mortality,
monitoring, and eradication modules into a single forward-time loop on
a 2-D structured grid.

Typical usage via the orchestrator::

    model = PopulationModel.from_config(config, habitat)
    result = model.run()

The returned *result* dict contains the density history (optionally
subsampled via ``snapshot_interval``), per-timestep logs from every
sub-model, and the grid coordinates.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from eradication.culling.model import EradicationModel
from eradication.monitoring.strategy import MonitoringModel
from eradication.population import init_point_source
from eradication.population.age_structure import AgeStructure
from eradication.population.dispersal import DispersalModel
from eradication.population.mortality import MortalityModel
from eradication.population.reproduction import GrowthModel
from copernicus_pipeline.interpolate import make_model_grid, build_model_timesteps

log = logging.getLogger(__name__)


class PopulationModel:
    """
    Spatially-explicit, age-structured population model.

    Parameters
    ----------
    config : dict
        Full scenario config (needs ``spatial``, ``temporal``, ``organism``,
        ``invasion``, ``monitoring``, ``eradication`` sections).
    habitat : xr.DataArray
        Habitat suitability mask — either ``(lat, lon)`` static or
        ``(time, lat, lon)`` time-varying.  dtype bool.
    monitoring : MonitoringModel
        Pre-built monitoring model.
    eradication : EradicationModel
        Pre-built eradication (culling) model.
    growth : GrowthModel
        Pre-built growth / recruitment model.
    dispersal : DispersalModel
        Pre-built near-field dispersal model.
    mortality : MortalityModel
        Pre-built natural mortality model.
    connectivity : dict | None
        Lagrangian connectivity tensor (unused in v1).
    snapshot_interval : int
        Store a density snapshot every *n* timesteps.  1 = every step.
    """

    def __init__(
        self,
        config: dict,
        habitat: xr.DataArray,
        monitoring: MonitoringModel,
        eradication: EradicationModel,
        growth: GrowthModel,
        dispersal: DispersalModel,
        mortality: MortalityModel,
        connectivity: dict | None = None,
        snapshot_interval: int = 1,
    ) -> None:
        # Grid coordinates
        self._lons, self._lats = make_model_grid(config["spatial"])
        ny, nx = len(self._lats), len(self._lons)

        # Habitat — static (lat, lon) or time-varying (time, lat, lon)
        if "time" in habitat.dims:
            self._habitat_static = False
            self._habitat_3d = habitat.values.astype(bool)  # (nt, ny, nx)
        else:
            self._habitat_static = True
            self._habitat_2d = habitat.values.astype(bool)  # (ny, nx)

        # Model timesteps
        self._timesteps = build_model_timesteps(config["temporal"])
        self.n_timesteps = len(self._timesteps)

        # Age structure
        self._ages = AgeStructure.from_config(config["organism"], ny, nx)

        # Seed initial population into the youngest age bin
        init_density = init_point_source(
            config["invasion"], config["spatial"], self._get_habitat_mask(0),
        )
        self._ages.add_recruits(init_density)

        # Sub-models
        self._growth = growth
        self._dispersal = dispersal
        self._mortality = mortality
        self._monitoring = monitoring
        self._eradication = eradication
        self._connectivity = connectivity  # placeholder for v2

        # Output
        self._snapshot_interval = max(1, snapshot_interval)
        self._snapshots: list[np.ndarray] = []
        self._snapshot_timesteps: list[int] = []
        self._log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the full simulation.

        Returns
        -------
        dict with keys:
            ``density_snapshots``  — (n_snaps, ny, nx) float32
            ``snapshot_timesteps`` — list[int]
            ``population_log``     — list[dict]
            ``monitoring_log``     — list[dict]
            ``eradication_log``    — list[dict]
            ``growth_log``         — list[dict]
            ``dispersal_log``      — list[dict]
            ``mortality_log``      — list[dict]
            ``lats``               — 1-D array
            ``lons``               — 1-D array
            ``timesteps``          — 1-D datetime64 array
        """
        log.info(
            "Starting population model: %d timesteps, grid %dx%d, %d age bins",
            self.n_timesteps, len(self._lats), len(self._lons), self._ages.n_ages,
        )

        for t in range(self.n_timesteps):
            self._step(t)

            if t % 52 == 0 or t == self.n_timesteps - 1:
                total = float(self._ages.total_density().sum())
                occ = self._ages.occupied_cells()
                log.info(
                    "  t=%4d  total_density=%.1f  occupied_cells=%d", t, total, occ,
                )

        return self._build_result()

    @property
    def log(self) -> list[dict[str, Any]]:
        return self._log

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict,
        habitat: xr.DataArray,
        connectivity: dict | None = None,
    ) -> "PopulationModel":
        """
        Build a fully-wired PopulationModel from the scenario config.

        Constructs monitoring, eradication, growth, dispersal, and
        mortality sub-models from their respective config sections.
        """
        organism_cfg = config["organism"]
        temporal_cfg = config["temporal"]
        dt_weeks = int(temporal_cfg["dt_weeks"])

        # Resolve a static habitat mask for strategies that need it at
        # construction time (monitoring strategies use it for cell pools).
        if "time" in habitat.dims:
            static_mask = habitat.values.astype(bool).all(axis=0)
        else:
            static_mask = habitat.values.astype(bool)

        monitoring = MonitoringModel.from_config(
            config["monitoring"],
            dt_weeks=dt_weeks,
            seed=0,
            habitat_mask=static_mask,
        )
        eradication = EradicationModel.from_config(
            config["eradication"],
            seed=100,
        )
        growth = GrowthModel.from_config(organism_cfg)
        dispersal = DispersalModel.from_config(organism_cfg)
        mortality = MortalityModel.from_config(organism_cfg)

        snapshot_interval = int(organism_cfg.get("snapshot_interval", 1))

        return cls(
            config=config,
            habitat=habitat,
            monitoring=monitoring,
            eradication=eradication,
            growth=growth,
            dispersal=dispersal,
            mortality=mortality,
            connectivity=connectivity,
            snapshot_interval=snapshot_interval,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_habitat_mask(self, timestep: int) -> np.ndarray:
        """
        Return the (ny, nx) bool habitat mask for the given timestep.

        For static habitat this always returns the same array.  For
        time-varying habitat the closest available time slice is used.
        """
        if self._habitat_static:
            return self._habitat_2d
        # Time-varying: clamp index to available range
        t_idx = min(timestep, self._habitat_3d.shape[0] - 1)
        return self._habitat_3d[t_idx]

    def _step(self, timestep: int) -> None:
        """
        Single timestep:
          1. Age (shift cohorts forward)
          2. Growth → recruits into age-bin 0
          3. Near-field dispersal
          4. Natural mortality
          5. Monitoring → detection response
          6. Eradication → cull efficiency
          7. Apply eradication mortality
          8. Enforce habitat mask
          9. Log & snapshot
        """
        habitat_mask = self._get_habitat_mask(timestep)

        # 1. Age — shift bins forward, oldest die, bin 0 cleared
        self._ages.age()

        # 2. Growth — recruits based on total density, placed into bin 0
        total = self._ages.total_density()
        recruits = self._growth.step(total, habitat_mask, timestep)
        self._ages.add_recruits(recruits)

        # 3. Dispersal — applied per age bin
        self._ages.density = self._dispersal.step(
            self._ages.density, habitat_mask, timestep,
        )

        # 4. Natural mortality
        self._ages.density = self._mortality.step(
            self._ages.density, timestep,
        )

        # 5. Monitoring
        total = self._ages.total_density()
        response = self._monitoring.step(total, timestep)

        # 6. Eradication
        cull_efficiency = self._eradication.step(response, timestep)

        # 7. Apply eradication mortality across all age bins
        self._ages.apply_mortality(cull_efficiency)

        # 8. Zero out unsuitable habitat
        self._ages.apply_habitat_mask(habitat_mask)

        # 9. Log
        total = self._ages.total_density()
        self._log.append({
            "timestep": timestep,
            "total_density": float(total.sum()),
            "occupied_cells": int((total > 0).sum()),
            "max_density": float(total.max()) if total.size > 0 else 0.0,
        })

        # 10. Snapshot
        if timestep % self._snapshot_interval == 0:
            self._snapshots.append(total.copy())
            self._snapshot_timesteps.append(timestep)

    def _build_result(self) -> dict:
        """Package simulation outputs into a result dict."""
        return {
            "density_snapshots": np.stack(self._snapshots) if self._snapshots else np.empty((0,)),
            "snapshot_timesteps": self._snapshot_timesteps,
            "population_log": self._log,
            "monitoring_log": self._monitoring.log,
            "eradication_log": self._eradication.log,
            "growth_log": self._growth.log,
            "dispersal_log": self._dispersal.log,
            "mortality_log": self._mortality.log,
            "lats": self._lats,
            "lons": self._lons,
            "timesteps": self._timesteps,
        }
