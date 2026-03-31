"""
Near-field dispersal models for the population simulation.

Each model receives the full age-structured density array
``(n_ages, ny, nx)`` and redistributes a fraction of each cell's
density to neighbouring cells.  The dispersal is applied independently
to every age bin.

One implementation is provided:

    GaussianNearFieldDispersal — small Gaussian kernel convolution via
                                 ``scipy.ndimage.convolve``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.ndimage import convolve

log = logging.getLogger(__name__)


class NearFieldDispersalModel(ABC):
    """Abstract base for near-field dispersal models."""

    def __init__(self) -> None:
        self._log: list[dict[str, Any]] = []

    @abstractmethod
    def step(
        self,
        density: np.ndarray,
        habitat_mask: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        """
        Apply dispersal to the age-structured density field.

        Parameters
        ----------
        density : (n_ages, ny, nx) float32 array
            Full age-structured density (modified in-place or returned).
        habitat_mask : (ny, nx) bool array
            True = suitable habitat.
        timestep : int
            Current model timestep index (0-based).

        Returns
        -------
        density : (n_ages, ny, nx) float32 array
            Updated density after dispersal.
        """

    @property
    def log(self) -> list[dict[str, Any]]:
        return self._log

    @classmethod
    def from_config(cls, organism_cfg: dict) -> "NearFieldDispersalModel":
        """
        Factory: build a near-field dispersal model from the ``organism`` config.

        If ``near_field_dispersal_sigma_cells`` is absent near-field
        dispersal is disabled (returns :class:`NoNearFieldDispersal`).
        """
        if "near_field_dispersal_sigma_cells" not in organism_cfg:
            log.warning(
                "Near-field dispersal disabled: 'near_field_dispersal_sigma_cells' not found in organism config."
            )
            return NoNearFieldDispersal()
        return GaussianNearFieldDispersal(
            sigma=float(organism_cfg["near_field_dispersal_sigma_cells"]),
            dispersal_fraction=float(organism_cfg["near_field_dispersal_fraction"]),
        )


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class NoNearFieldDispersal(NearFieldDispersalModel):
    """
    No-op near-field dispersal — density unchanged each timestep.

    Selected automatically when ``near_field_dispersal_sigma_cells`` is
    absent from the organism config.
    """

    def step(
        self,
        density: np.ndarray,
        habitat_mask: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        self._log.append({"timestep": timestep, "total_dispersed": 0.0})
        return density


class GaussianNearFieldDispersal(NearFieldDispersalModel):
    """
    Near-field dispersal using a small 2-D Gaussian kernel.

    Each timestep, a fraction (``dispersal_fraction``) of each cell's
    density is removed and redistributed to the 3x3 neighbourhood
    according to a Gaussian weighting.  The remainder stays in place.

    The kernel is precomputed once at construction and reused.
    ``scipy.ndimage.convolve`` with ``mode='constant', cval=0.0`` is
    used — density that disperses beyond grid edges or into unsuitable
    habitat is lost.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel in grid cells.
    dispersal_fraction : float
        Fraction of each cell's density that enters the dispersal pool
        per timestep.  Must be in (0, 1].
    """

    KERNEL_RADIUS: int = 1  # 3x3 kernel

    def __init__(self, sigma: float, dispersal_fraction: float) -> None:
        super().__init__()
        if not (0.0 < dispersal_fraction <= 1.0):
            raise ValueError("dispersal_fraction must be in (0, 1]")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.sigma = sigma
        self.dispersal_fraction = dispersal_fraction
        self._kernel = self._build_kernel(sigma, self.KERNEL_RADIUS)

    @staticmethod
    def _build_kernel(sigma: float, radius: int) -> np.ndarray:
        """
        Build a normalised 2-D Gaussian kernel with the centre zeroed.

        The centre is excluded because the staying fraction is handled
        separately (``1 - dispersal_fraction``).  The remaining weights
        are re-normalised so they sum to 1.
        """
        size = 2 * radius + 1
        ax = np.arange(size) - radius
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel[radius, radius] = 0.0
        total = kernel.sum()
        if total > 0:
            kernel /= total
        return kernel.astype(np.float32)

    def step(
        self,
        density: np.ndarray,
        habitat_mask: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        n_ages = density.shape[0]
        total_dispersed = 0.0

        for a in range(n_ages):
            layer = density[a]
            dispersing = layer * self.dispersal_fraction
            staying = layer * (1.0 - self.dispersal_fraction)

            spread = convolve(
                dispersing, self._kernel, mode="constant", cval=0.0,
            )

            new_layer = staying + spread
            new_layer[~habitat_mask] = 0.0
            np.clip(new_layer, 0.0, None, out=new_layer)
            density[a] = new_layer.astype(np.float32)

            total_dispersed += float(dispersing.sum())

        self._log.append({
            "timestep": timestep,
            "total_dispersed": total_dispersed,
        })
        return density
