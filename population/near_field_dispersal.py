"""
Near-field dispersal models for the population simulation.

Each model receives the full age-structured density array
``(n_ages, ny, nx)`` and redistributes a fraction of each cell's
density to neighbouring cells.  The dispersal is applied independently
to every age bin.

## Organism-type behaviour

The operation performed on each age bin differs by organism type:

* **discrete** — each cell's individuals independently decide to disperse
  via ``Binomial(N, dispersal_fraction)``.  Dispersing individuals are
  then distributed to valid neighbours by Multinomial sampling weighted
  by the Gaussian kernel.  Individuals that would disperse beyond the
  grid boundary are lost.  All counts remain integers.

* **continuous** — standard Gaussian convolution on the coverage fraction
  array (current behaviour, unchanged).

One implementation is provided:

    GaussianNearFieldDispersal — small Gaussian kernel convolution
                                 (continuous) or Binomial + Multinomial
                                 routing (discrete).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.ndimage import convolve

log = logging.getLogger(__name__)


def _scatter_multinomial(
    n_leaving: np.ndarray,
    kernel: np.ndarray,
    rng: np.random.Generator,
    ny: int,
    nx: int,
) -> np.ndarray:
    """
    Distribute integer counts from each cell to valid neighbours via
    Multinomial sampling weighted by the Gaussian kernel.

    For each occupied cell (y, x) the ``n_leaving[y, x]`` individuals are
    split into two stages:

    1. ``Binomial(n_leaving, p_valid)`` determines how many reach a valid
       destination (rather than being absorbed by the grid boundary).
    2. ``Multinomial(n_in_grid, normalised_neighbour_weights)`` routes the
       survivors among the valid neighbour cells.

    Parameters
    ----------
    n_leaving : (ny, nx) int64 array
        Number of individuals leaving each cell this timestep.
    kernel : (3, 3) float32 array
        Gaussian kernel with centre zeroed and off-centre weights summing
        to 1 for interior cells.
    rng : np.random.Generator
        Shared random-number generator.
    ny, nx : int
        Grid dimensions.

    Returns
    -------
    spread : (ny, nx) int64 array
        Number of individuals arriving at each cell from neighbours.

    Notes
    -----
    Time complexity: O(n_occupied) Multinomial draws per call, where
    n_occupied is the number of non-zero cells in ``n_leaving``.  This is
    efficient for sparse populations and acceptable for dense ones.
    """
    spread = np.zeros((ny, nx), dtype=np.int64)

    # Precompute (dy, dx, weight) for the 8 off-centre kernel positions
    offsets: list[tuple[int, int, float]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            offsets.append((dy, dx, float(kernel[1 + dy, 1 + dx])))

    occupied_ys, occupied_xs = np.nonzero(n_leaving)

    for y, x in zip(occupied_ys.tolist(), occupied_xs.tolist()):
        n = int(n_leaving[y, x])
        if n == 0:
            continue

        # Build lists of valid destination cells and their weights
        dests: list[tuple[int, int]] = []
        probs: list[float] = []
        for dy, dx, w in offsets:
            dst_y, dst_x = y + dy, x + dx
            if 0 <= dst_y < ny and 0 <= dst_x < nx:
                dests.append((dst_y, dst_x))
                probs.append(w)

        if not dests:
            continue

        p_valid = sum(probs)
        if p_valid <= 0.0:
            continue

        # Stage 1: how many individuals reach a valid cell (boundary loss)
        n_in_grid = int(rng.binomial(n, min(p_valid, 1.0)))
        if n_in_grid == 0:
            continue

        # Stage 2: route survivors among valid neighbours
        probs_arr = np.array(probs, dtype=np.float64) / p_valid
        counts = rng.multinomial(n_in_grid, probs_arr)
        for (dst_y, dst_x), cnt in zip(dests, counts):
            spread[dst_y, dst_x] += cnt

    return spread


class NearFieldDispersalModel(ABC):
    """Abstract base for near-field dispersal models."""

    def __init__(self) -> None:
        self._log: list[dict[str, Any]] = []

    @abstractmethod
    def compute_incoming(self, density: np.ndarray) -> np.ndarray:
        """
        Compute the total near-field incoming flux per cell without
        modifying the density array.

        Used to build the potential-growth buffer before suppression is
        applied.  Only the *incoming spread* is returned (the outgoing
        fraction is a loss, not growth, and is not suppressed).

        Parameters
        ----------
        density : (n_ages, ny, nx) float32 array
            Current age-structured state.
            Units: individuals/cell (discrete) | coverage [0-1] (continuous).

        Returns
        -------
        incoming : (ny, nx) float32 array
            Total incoming spread summed across all age bins.
        """

    @abstractmethod
    def step(
        self,
        density: np.ndarray,
        habitat_mask: np.ndarray,
        timestep: int,
        suppression: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply dispersal to the age-structured density field.

        Parameters
        ----------
        density : (n_ages, ny, nx) float32 array
            Full age-structured state (modified in-place or returned).
            Units: individuals/cell (discrete) | coverage [0-1] (continuous).
        habitat_mask : (ny, nx) bool array
            True = suitable habitat.
        timestep : int
            Current model timestep index (0-based).
        suppression : (ny, nx) float32 array or None
            Per-cell suppression factor in [0, 1].  When provided the
            incoming spread at each cell is multiplied by this factor
            before being added to the staying fraction.  The outgoing
            fraction is unaffected.

        Returns
        -------
        density : (n_ages, ny, nx) float32 array
            Updated density after dispersal.
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
    ) -> "NearFieldDispersalModel":
        """
        Factory: build a near-field dispersal model from the ``organism`` config.

        If ``near_field_dispersal_sigma_cells`` is absent near-field
        dispersal is disabled (returns :class:`NoNearFieldDispersal`).

        Parameters
        ----------
        organism_cfg : dict
            ``config["organism"]`` section.
        organism_type : str
            ``"discrete"`` or ``"continuous"`` — controls the dispersal
            algorithm used in ``step()``.
        rng : np.random.Generator, optional
            Shared RNG for Binomial/Multinomial draws (discrete mode only).
        """
        if "near_field_dispersal_sigma_cells" not in organism_cfg:
            log.warning(
                "Near-field dispersal disabled: 'near_field_dispersal_sigma_cells' not found in organism config."
            )
            return NoNearFieldDispersal()
        return GaussianNearFieldDispersal(
            sigma=float(organism_cfg["near_field_dispersal_sigma_cells"]),
            dispersal_fraction=float(organism_cfg["near_field_dispersal_fraction"]),
            organism_type=organism_type,
            rng=rng,
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

    def compute_incoming(self, density: np.ndarray) -> np.ndarray:
        ny, nx = density.shape[1], density.shape[2]
        return np.zeros((ny, nx), dtype=np.float32)

    def step(
        self,
        density: np.ndarray,
        habitat_mask: np.ndarray,
        timestep: int,
        suppression: np.ndarray | None = None,
    ) -> np.ndarray:
        self._log.append({"timestep": timestep, "total_dispersed": 0.0})
        return density


class GaussianNearFieldDispersal(NearFieldDispersalModel):
    """
    Near-field dispersal using a small 2-D Gaussian kernel.

    Each timestep, a fraction (``dispersal_fraction``) of each cell's
    density is removed and redistributed to the 3*3 neighbourhood
    according to a Gaussian weighting.  The remainder stays in place.

    The operation depends on ``organism_type``:

    * **discrete** — ``Binomial(N, dispersal_fraction)`` determines how many
      individuals leave each cell.  Those individuals are then routed to
      valid neighbours via ``Multinomial`` sampling weighted by the kernel.
      All counts remain integers.  Boundary loss is explicit.
    * **continuous** — ``scipy.ndimage.convolve`` with
      ``mode='constant', cval=0.0`` applied to the coverage fraction array.
      Density dispersing beyond grid edges or into unsuitable habitat is lost.

    The kernel is precomputed once at construction and reused.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel in grid cells.
    dispersal_fraction : float
        Fraction of each cell's density that enters the dispersal pool
        per timestep.  Must be in (0, 1].
    organism_type : str
        ``"discrete"`` or ``"continuous"``.
    rng : np.random.Generator, optional
        Shared RNG for Binomial/Multinomial draws (discrete mode only).
    """

    KERNEL_RADIUS: int = 1  # 3x3 kernel

    def __init__(
        self,
        sigma: float,
        dispersal_fraction: float,
        organism_type: str = "discrete",
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        if not (0.0 < dispersal_fraction <= 1.0):
            raise ValueError("dispersal_fraction must be in (0, 1]")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if organism_type not in ("discrete", "continuous"):
            raise ValueError(
                f"organism_type must be 'discrete' or 'continuous', got {organism_type!r}."
            )
        self.sigma = sigma
        self.dispersal_fraction = dispersal_fraction
        self._organism_type = organism_type
        self._rng = rng if rng is not None else np.random.default_rng()
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

    def compute_incoming(self, density: np.ndarray) -> np.ndarray:
        """
        Estimate the expected near-field incoming flux per cell.

        For both organism types the expected incoming flux is approximated
        using Gaussian convolution on the current density (the deterministic
        expectation).  This is used only for carrying-capacity suppression
        pre-computation — the stochastic discrete draws happen in ``step()``.
        """
        ny, nx = density.shape[1], density.shape[2]
        incoming = np.zeros((ny, nx), dtype=np.float32)
        for a in range(density.shape[0]):
            dispersing = density[a] * self.dispersal_fraction
            incoming += convolve(dispersing, self._kernel, mode="constant", cval=0.0)
        return incoming

    def step(
        self,
        density: np.ndarray,
        habitat_mask: np.ndarray,
        timestep: int,
        suppression: np.ndarray | None = None,
    ) -> np.ndarray:
        n_ages = density.shape[0]
        ny, nx = density.shape[1], density.shape[2]
        total_dispersed = 0.0

        for a in range(n_ages):
            layer = density[a]

            if self._organism_type == "discrete":
                # Binomial: each individual independently decides to disperse
                n_layer = layer.astype(np.int64)
                n_leaving = self._rng.binomial(n_layer, self.dispersal_fraction)
                staying = (n_layer - n_leaving).astype(np.float32)

                # Multinomial: route leavers to valid neighbours
                spread = _scatter_multinomial(
                    n_leaving, self._kernel, self._rng, ny, nx,
                ).astype(np.float32)

                if suppression is not None:
                    spread = spread * suppression

                new_layer = staying + spread

            else:  # "continuous"
                dispersing = layer * self.dispersal_fraction
                staying = layer * (1.0 - self.dispersal_fraction)

                spread = convolve(dispersing, self._kernel, mode="constant", cval=0.0)
                if suppression is not None:
                    spread = spread * suppression

                new_layer = staying + spread

            new_layer[~habitat_mask] = 0.0
            np.clip(new_layer, 0.0, None, out=new_layer)
            density[a] = new_layer.astype(np.float32)

            total_dispersed += float(layer.sum() * self.dispersal_fraction)

        self._log.append({
            "timestep": timestep,
            "total_dispersed": total_dispersed,
        })
        return density
