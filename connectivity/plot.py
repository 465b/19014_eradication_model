"""
Debug visualisation for the Lagrangian connectivity tensor.

plot_sink_connectivity(connectivity, lons, lats, out_dir)
    Map of total incoming weight at each grid cell, summed over all
    source cells and age bins.  Answers: "how much water from anywhere
    can reach each location?"
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

log = logging.getLogger(__name__)

_UNSUITABLE_COLOR = "#b0b0b0"
_CARTOPY_RESOLUTION = "10m"


def plot_sink_connectivity(
    connectivity: dict,
    lons: np.ndarray,
    lats: np.ndarray,
    out_dir: Path,
    habitat_mask: np.ndarray | None = None,
    filename: str = "debug_sink_connectivity.png",
) -> Path:
    """
    Save a map of total incoming connectivity weight at each grid cell.

    For every entry in the connectivity tensor the destination cell
    receives ``weight``.  Summing over all sources and age bins gives a
    2-D field that shows which cells are most reachable from the full
    set of sources — the "sink connectivity".

    Parameters
    ----------
    connectivity : dict
        As returned by :func:`eradication.connectivity.load.load_connectivity`.
        Must contain ``dst_x`` (ix), ``dst_y`` (iy), and ``weight``.
    lons : 1-D array
        Grid longitude centres (nx,).
    lats : 1-D array
        Grid latitude centres (ny,).
    out_dir : Path
        Directory for the output PNG (created if absent).
    habitat_mask : (ny, nx) bool array, optional
        Unsuitable cells are shown in grey.
    filename : str
        Output filename.

    Returns
    -------
    Path to the saved PNG.
    """
    ny, nx = len(lats), len(lons)

    # --- Accumulate weights at each destination cell ---
    sink_map = np.zeros((ny, nx), dtype=np.float32)
    dst_x = connectivity["dst_x"].astype(np.intp)
    dst_y = connectivity["dst_y"].astype(np.intp)
    weight = connectivity["weight"]

    valid = (dst_x >= 0) & (dst_x < nx) & (dst_y >= 0) & (dst_y < ny)
    np.add.at(sink_map, (dst_y[valid], dst_x[valid]), weight[valid])

    # --- Mask unsuitable cells ---
    display = sink_map.astype(float)
    if habitat_mask is not None:
        display[~habitat_mask] = np.nan

    vmax = float(np.nanmax(display)) if np.any(np.isfinite(display)) else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.cm.Blues

    # --- Figure ---
    if _HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": proj})
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Grey layer for unsuitable habitat
    if habitat_mask is not None:
        grey_data = np.where(habitat_mask, np.nan, 0.0)
        grey_cmap = mcolors.ListedColormap([_UNSUITABLE_COLOR])
        kw_grey = dict(zorder=1)
        if _HAS_CARTOPY:
            kw_grey["transform"] = ccrs.PlateCarree()
        ax.pcolormesh(lons, lats, grey_data,
                      cmap=grey_cmap, vmin=-0.5, vmax=0.5, **kw_grey)

    # Sink connectivity layer
    kw = dict(norm=norm, cmap=cmap, zorder=2)
    if _HAS_CARTOPY:
        kw["transform"] = ccrs.PlateCarree()
        kw["zorder"] = 3
        ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]],
                      crs=ccrs.PlateCarree())
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "coastline", _CARTOPY_RESOLUTION,
                edgecolor="black", facecolor="none",
            ),
            linewidth=0.6, zorder=4,
        )
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    else:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    mesh = ax.pcolormesh(lons, lats, display, **kw)

    cb = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("total incoming weight  [summed over all sources & ages]", fontsize=8)

    n_entries = int(valid.sum())
    n_sources = int(len(np.unique(
        np.stack([connectivity["src_x"][valid], connectivity["src_y"][valid]], axis=1),
        axis=0,
    ))) if n_entries > 0 else 0
    ax.set_title(
        f"Sink connectivity  ({n_sources} sources, {n_entries:,} entries)",
        fontsize=10,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Sink connectivity plot saved → %s", out_path)
    return out_path
