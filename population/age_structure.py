"""
Age-structured population state on a 2D spatial grid.

Manages a 3-D density array of shape ``(n_ages, ny, nx)`` where each slice
along axis 0 is one weekly age cohort.  Provides the core bookkeeping:

    age()               — shift all cohorts forward by one timestep (oldest die)
    add_recruits(r)     — place new individuals into age-bin 0
    total_density()     — sum across ages → (ny, nx)
    apply_mortality(m)  — element-wise removal across all age bins

The number of age bins equals ``organism_cfg["max_age_weeks"]``.
Every bin has equal width (one model timestep = one week).
"""

from __future__ import annotations

import numpy as np


class AgeStructure:
    """
    Dense age-structured population state.

    Parameters
    ----------
    n_ages : int
        Total number of age bins (each spanning one model timestep).
    ny, nx : int
        Spatial grid dimensions (lat, lon).
    """

    def __init__(self, n_ages: int, ny: int, nx: int) -> None:
        if n_ages < 1:
            raise ValueError("n_ages must be >= 1")
        self.n_ages = n_ages
        self.ny = ny
        self.nx = nx
        self.density = np.zeros((n_ages, ny, nx), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def age(self) -> None:
        """
        Advance all cohorts by one timestep.

        Individuals in the oldest bin (index ``n_ages - 1``) are removed
        (die of old age).  Every other bin shifts forward by one index.
        Bin 0 is zeroed — the caller is responsible for filling it via
        :meth:`add_recruits`.
        """
        # Roll forward: bin[i] → bin[i+1], bin[0] ← 0
        # np.roll wraps around, so we zero the oldest bin first, then roll.
        self.density[-1] = 0.0
        self.density = np.roll(self.density, shift=1, axis=0)
        # bin 0 now contains what was in the last (zeroed) bin → already 0

    def add_recruits(self, recruits: np.ndarray) -> None:
        """
        Place new recruits into the youngest age bin.

        Parameters
        ----------
        recruits : (ny, nx) float32 array
            Density of new individuals to add to age bin 0.
        """
        self.density[0] += recruits

    def total_density(self) -> np.ndarray:
        """
        Sum density across all age bins.

        Returns
        -------
        (ny, nx) float32 array
        """
        return self.density.sum(axis=0)

    def apply_mortality(self, cull_efficiency: np.ndarray) -> None:
        """
        Apply a spatially-varying mortality field to every age bin.

        Parameters
        ----------
        cull_efficiency : (ny, nx) float32 array
            Fraction of individuals to remove per cell (0 = no removal,
            1 = complete removal).  Applied identically to all age bins.
        """
        survival = 1.0 - cull_efficiency
        self.density *= survival[np.newaxis, :, :]

    def apply_habitat_mask(self, habitat_mask: np.ndarray) -> None:
        """
        Zero out density in cells that are not suitable habitat.

        Parameters
        ----------
        habitat_mask : (ny, nx) bool array
            True = suitable, False = unsuitable.
        """
        self.density[:, ~habitat_mask] = 0.0

    def occupied_cells(self) -> int:
        """Number of cells with non-zero total density."""
        return int((self.total_density() > 0).sum())

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, organism_cfg: dict, ny: int, nx: int) -> "AgeStructure":
        """
        Build an AgeStructure from the ``organism`` config section.

        ``n_ages`` equals ``organism_cfg["max_age_weeks"]``.
        Each bin spans one model timestep (one week).
        """
        n_ages = int(organism_cfg["max_age_weeks"])
        return cls(n_ages=n_ages, ny=ny, nx=nx)
