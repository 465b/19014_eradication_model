"""
Population initialisation helpers.

Provides factory functions that create the initial density field for the
population model based on the ``invasion`` config section.
"""

from __future__ import annotations

import numpy as np

from copernicus_pipeline.interpolate import make_model_grid


def init_point_source(
    invasion_cfg: dict,
    spatial_cfg: dict,
    habitat_mask: np.ndarray,
) -> np.ndarray:
    """
    Create an initial density field by seeding a single grid cell.

    Finds the cell nearest to ``(invasion.location.x, invasion.location.y)``,
    verifies it is suitable habitat, and sets its density to
    ``invasion.initial_density``.

    Parameters
    ----------
    invasion_cfg : dict
        The ``invasion`` section of the scenario YAML.  Required keys:
        ``type`` (must be ``"point_source"``), ``location.x`` (longitude),
        ``location.y`` (latitude), ``initial_density``.
    spatial_cfg : dict
        The ``spatial`` section of the scenario YAML (used to reconstruct
        the model grid).
    habitat_mask : (ny, nx) bool array
        Suitable-habitat mask.  True = suitable.

    Returns
    -------
    density : (ny, nx) float32 array
        All zeros except the seeded cell.

    Raises
    ------
    ValueError
        If the invasion type is not ``"point_source"`` or the nearest cell
        is not suitable habitat.
    """
    if invasion_cfg["type"] != "point_source":
        raise ValueError(
            f"Unknown invasion type {invasion_cfg['type']!r}. "
            "Only 'point_source' is supported."
        )

    lons, lats = make_model_grid(spatial_cfg)
    ny, nx = len(lats), len(lons)

    loc_x = float(invasion_cfg["location"]["x"])  # longitude
    loc_y = float(invasion_cfg["location"]["y"])  # latitude

    j = int(np.argmin(np.abs(lats - loc_y)))  # lat index (row)
    i = int(np.argmin(np.abs(lons - loc_x)))  # lon index (col)

    if not habitat_mask[j, i]:
        raise ValueError(
            f"Invasion point ({loc_x}, {loc_y}) maps to grid cell "
            f"[{j}, {i}] which is not suitable habitat."
        )

    density = np.zeros((ny, nx), dtype=np.float32)
    density[j, i] = float(invasion_cfg["initial_density"])
    return density
