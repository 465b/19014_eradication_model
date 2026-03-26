"""
eradication/habitat/schism_depth.py
-------------------------------------
Read SCHISM unstructured-grid NetCDF files and interpolate depth data
onto the eradication model's regular lat/lon grid.

SCHISM stores its mesh nodes in a projected CRS (default: NZTM2000 / EPSG:2193).
This module handles the CRS conversion and interpolation — using the mesh
topology from SCHISM_hgrid_face_nodes — onto the model's regular lat/lon grid.

Public API
----------
schism_depth_to_grid(path, mode, crs, model_lons, model_lats) -> np.ndarray
    Returns a float32 array of shape (n_lat, n_lon).
    Node depth values are interpolated using matplotlib.tri.LinearTriInterpolator
    over the SCHISM mesh triangulation, with a NearestNDInterpolator fallback
    for cells outside the triangulated domain.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

_NODE_X     = "SCHISM_hgrid_node_x"
_NODE_Y     = "SCHISM_hgrid_node_y"
_FACE_NODES = "SCHISM_hgrid_face_nodes"
_DEPTH_VAR  = "depth"
_ELEV_VAR   = "elev"

_DEFAULT_CRS    = "EPSG:2193"   # NZTM2000 — standard for NZ SCHISM runs
_DOMAIN_PAD_DEG = 0.5           # degrees of padding around model bbox for triangle subsetting


def schism_depth_to_grid(
    path: Path,
    mode: str,
    crs: str,
    model_lons: np.ndarray,
    model_lats: np.ndarray,
) -> np.ndarray:
    """Interpolate SCHISM depth data onto the regular model lat/lon grid.

    Parameters
    ----------
    path : Path
        Path to a SCHISM output NetCDF file.
    mode : str
        ``"depth_below_geoid"``
            Use the static ``depth`` variable (bathymetry below mean sea level).
        ``"total_water_depth"``
            Use ``depth + mean(elev, axis=0)``, giving the tidal-mean total
            water column thickness.
    crs : str
        PROJ/EPSG identifier for the CRS of the SCHISM node x/y coordinates.
        Default is ``"EPSG:2193"`` (NZTM2000).
    model_lons : np.ndarray
        1-D array of model grid cell centre longitudes (WGS84 degrees).
    model_lats : np.ndarray
        1-D array of model grid cell centre latitudes (WGS84 degrees).

    Returns
    -------
    np.ndarray
        float32 array of shape (n_lat, n_lon) with depth values in metres,
        positive down, interpolated onto the model grid.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If a required variable is missing from the file.
    ValueError
        If no SCHISM triangles fall within the padded model domain (check *crs*),
        or if *mode* is not recognised.
    """
    from pyproj import Transformer
    from scipy.interpolate import NearestNDInterpolator
    import matplotlib.tri as mtri

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SCHISM file not found: {path}")

    if mode not in ("depth_below_geoid", "total_water_depth"):
        raise ValueError(
            f"Unknown schism_depth_constraints mode: '{mode}'. "
            "Expected 'depth_below_geoid' or 'total_water_depth'."
        )

    with xr.open_dataset(path, mask_and_scale=True) as ds:
        # --- variable presence checks ---
        for required in (_NODE_X, _NODE_Y, _FACE_NODES, _DEPTH_VAR):
            if required not in ds:
                raise KeyError(
                    f"Variable '{required}' not found in SCHISM file {path}. "
                    f"Available variables: {sorted(ds.data_vars) + sorted(ds.coords)}"
                )
        if mode == "total_water_depth" and _ELEV_VAR not in ds:
            raise KeyError(
                f"Variable '{_ELEV_VAR}' required for mode 'total_water_depth' "
                f"but not found in {path}. Available variables: "
                f"{sorted(ds.data_vars) + sorted(ds.coords)}"
            )

        node_x = ds[_NODE_X].values.astype(float)    # projected x (metres)
        node_y = ds[_NODE_Y].values.astype(float)    # projected y (metres)
        depth  = ds[_DEPTH_VAR].values.astype(float) # (n_nodes,)

        # Take only the first 3 node indices per face (triangles; quads use 4
        # but the 4th column is NaN).  Convert 1-based UGRID indices to 0-based.
        face_nodes = ds[_FACE_NODES].values[:, :3].astype(np.int64) - 1  # (n_faces, 3)

        if mode == "total_water_depth":
            elev_mean = ds[_ELEV_VAR].values.astype(float).mean(axis=0)  # (n_nodes,)
            depth_values = depth + elev_mean
        else:
            depth_values = depth

        logger.debug("SCHISM: loaded %d nodes, %d triangles from %s",
                     len(node_x), len(face_nodes), path)

    # --- CRS conversion: projected metres → WGS84 lon/lat ---
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    node_lons, node_lats = transformer.transform(node_x, node_y)

    # --- subset to padded model domain ---
    lon_min = model_lons.min() - _DOMAIN_PAD_DEG
    lon_max = model_lons.max() + _DOMAIN_PAD_DEG
    lat_min = model_lats.min() - _DOMAIN_PAD_DEG
    lat_max = model_lats.max() + _DOMAIN_PAD_DEG

    # Keep triangles whose three nodes all fall within the padded domain.
    # The 0.5° pad ensures triangles straddling the true model boundary are included.
    in_pad = (
        (node_lons >= lon_min) & (node_lons <= lon_max) &
        (node_lats >= lat_min) & (node_lats <= lat_max)
    )
    tri_mask = in_pad[face_nodes].all(axis=1)
    n_tris = int(tri_mask.sum())

    if n_tris == 0:
        raise ValueError(
            f"No SCHISM triangles found within model domain ±{_DOMAIN_PAD_DEG}° "
            f"(lon {lon_min:.3f}–{lon_max:.3f}, lat {lat_min:.3f}–{lat_max:.3f}). "
            f"Check that crs='{crs}' matches the file's coordinate system."
        )

    local_faces = face_nodes[tri_mask]   # (n_tris, 3), still global node indices

    # Remap global node indices to contiguous local indices so Triangulation
    # receives a compact node array.
    used_nodes = np.unique(local_faces)
    remap = np.empty(len(node_lons), dtype=np.intp)
    remap[used_nodes] = np.arange(len(used_nodes))
    local_triangles = remap[local_faces]   # (n_tris, 3), 0-based local indices

    logger.debug(
        "SCHISM: %d triangles, %d nodes in padded domain [%.3f–%.3f, %.3f–%.3f]",
        n_tris, len(used_nodes), lon_min, lon_max, lat_min, lat_max,
    )

    # --- interpolation using SCHISM mesh topology ---
    triang = mtri.Triangulation(
        node_lons[used_nodes],
        node_lats[used_nodes],
        local_triangles,
    )
    interp_tri = mtri.LinearTriInterpolator(triang, depth_values[used_nodes])

    mesh_lon, mesh_lat = np.meshgrid(model_lons, model_lats)
    result = np.asarray(interp_tri(mesh_lon, mesh_lat).filled(np.nan))

    # NaN fallback for cells outside the triangulated domain (e.g. near domain edge)
    nan_mask = np.isnan(result)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        pts_near = np.stack([node_lons[used_nodes], node_lats[used_nodes]], axis=-1)
        interp_near = NearestNDInterpolator(pts_near, depth_values[used_nodes])
        query = np.stack([mesh_lon[nan_mask], mesh_lat[nan_mask]], axis=-1)
        result[nan_mask] = interp_near(query)
        logger.debug("SCHISM: NearestND fallback filled %d / %d cells", n_nan, result.size)

    return result.astype(np.float32)
