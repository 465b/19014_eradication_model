"""
Visualisation of population model results.

Public API
----------
plot_spatial_snapshots(result, out_dir)
    N-panel map of density at evenly-spaced simulation timesteps.
    Unsuitable habitat is shown in grey; suitable-but-empty cells are white.

plot_time_series(result, out_dir)
    4-panel time series: total abundance, occupied cells,
    monitoring detections, and eradicated cells.

plot_animation(result, out_dir)
    MP4 (or GIF fallback) animation of density evolving over all snapshots.

plot_all(result, out_dir)
    Convenience wrapper — calls all three functions above.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# Cartopy is optional — plain pcolormesh is used as a fallback.
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

log = logging.getLogger(__name__)

_UNSUITABLE_COLOR = "#b0b0b0"   # grey for cells outside the habitat mask
_HABITAT_ZERO_COLOR = "white"   # suitable cells with zero density
_CARTOPY_RESOLUTION = "10m"     # Natural Earth resolution: "10m" | "50m" | "110m"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_all(result: dict, out_dir: Path) -> list[Path]:
    """
    Save all standard population plots to *out_dir*.

    Parameters
    ----------
    result : dict
        Return value of :meth:`PopulationModel.run`.
    out_dir : Path
        Directory for output files (created if absent).

    Returns
    -------
    list of Paths written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    written.append(plot_spatial_snapshots(result, out_dir))
    written.append(plot_time_series(result, out_dir))
    written.append(plot_animation(result, out_dir))
    return written


def plot_spatial_snapshots(
    result: dict,
    out_dir: Path,
    n_panels: int = 6,
    filename: str = "population_spatial_snapshots.png",
) -> Path:
    """
    N-panel map of total density at evenly-spaced simulation timesteps.

    Rendering:
      - Unsuitable habitat  → grey (``_UNSUITABLE_COLOR``)
      - Suitable, density=0 → white
      - Suitable, density>0 → inferno colormap, linear scale shared
                              across all panels

    Coastlines are drawn when cartopy is available.

    Parameters
    ----------
    result : dict
        Return value of :meth:`PopulationModel.run`.
    out_dir : Path
        Output directory.
    n_panels : int
        Number of time panels (default 6).
    filename : str
        Output filename (PNG).

    Returns
    -------
    Path to the saved PNG.
    """
    snaps        = result["density_snapshots"]          # (n_snaps, ny, nx)
    t_idx        = result["snapshot_timesteps"]         # list[int]
    lats         = result["lats"]
    lons         = result["lons"]
    tstamps      = result["timesteps"]                  # datetime64 array
    habitat_mask = result.get("habitat_mask")           # (ny, nx) bool or None

    n_snaps = snaps.shape[0]
    if n_snaps == 0:
        raise ValueError("No snapshots in result.")

    # Evenly-spaced panels
    chosen = np.round(np.linspace(0, n_snaps - 1, min(n_panels, n_snaps))).astype(int)

    # Shared linear colour scale across all panels (only over suitable cells)
    if habitat_mask is not None:
        suitable_snaps = snaps[:, habitat_mask]
    else:
        suitable_snaps = snaps
    vmax = float(suitable_snaps.max()) if suitable_snaps.max() > 0 else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    # Colourmap: white at 0, inferno palette for positive values
    cmap = _make_density_cmap()

    n_cols = min(3, len(chosen))
    n_rows = int(np.ceil(len(chosen) / n_cols))

    if _HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            subplot_kw={"projection": proj},
        )
    else:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
        )

    axes_flat = np.array(axes).ravel()

    for panel_i, snap_i in enumerate(chosen):
        ax = axes_flat[panel_i]
        _draw_density_map(
            ax, snaps[snap_i], lons, lats, habitat_mask, norm, cmap,
        )

        ts = t_idx[snap_i]
        date_str = str(tstamps[ts])[:10] if ts < len(tstamps) else f"t={ts}"
        ax.set_title(f"t = {date_str}", fontsize=9)

    for ax in axes_flat[len(chosen):]:
        ax.set_visible(False)

    # Shared colourbar
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.12)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("density  [ind / m²]", fontsize=9)

    fig.suptitle("Population density — spatial evolution", fontsize=11, y=1.01)

    out_path = Path(out_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Spatial snapshot plot saved → %s", out_path)
    return out_path


def plot_time_series(
    result: dict,
    out_dir: Path,
    filename: str = "population_time_series.png",
) -> Path:
    """
    4-panel time series of key population and management metrics.

    Panels:
      1. Total abundance (sum of density across all cells)
      2. Occupied cells (y-axis capped at total suitable habitat cells)
      3. Monitoring: detections and treatment responses per timestep
      4. Eradicated (treated) cells per timestep

    Parameters
    ----------
    result : dict
        Return value of :meth:`PopulationModel.run`.
    out_dir : Path
        Output directory.
    filename : str
        Output filename (PNG).

    Returns
    -------
    Path to the saved PNG.
    """
    pop_log  = result["population_log"]
    mon_log  = result["monitoring_log"]
    erad_log = result["eradication_log"]
    tstamps  = result["timesteps"]

    use_dates = len(tstamps) >= len(pop_log)

    def _xvals(log_list: list[dict]) -> list:
        ts = [e["timestep"] for e in log_list]
        return [tstamps[t] for t in ts] if use_dates else ts

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    xs = _xvals(pop_log)

    # --- Panel 1: total abundance ---
    ax = axes[0]
    ax.plot(xs, [e["total_density"] for e in pop_log],
            color="#1f77b4", linewidth=1.0)
    ax.set_ylabel("total abundance\n[ind / m²]", fontsize=8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # --- Panel 2: occupied cells ---
    ax = axes[1]
    ax.plot(xs, [e["occupied_cells"] for e in pop_log],
            color="#2ca02c", linewidth=1.0)
    ax.set_ylabel("occupied cells", fontsize=8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    n_habitat = result.get("n_habitat_cells")
    if n_habitat:
        ax.set_ylim(0, n_habitat)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # --- Panel 3: monitoring detections ---
    ax = axes[2]
    xs_mon = _xvals(mon_log)
    bw = _bar_width(xs_mon)
    ax.bar(xs_mon, [e["n_detected"]  for e in mon_log],
           label="detected",  color="#ff7f0e", alpha=0.7, width=bw)
    ax.bar(xs_mon, [e["n_responded"] for e in mon_log],
           label="responded", color="#d62728", alpha=0.9, width=bw)
    ax.set_ylabel("cells", fontsize=8)
    ax.set_title("monitoring", fontsize=8, loc="left")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # --- Panel 4: eradicated cells ---
    ax = axes[3]
    xs_er = _xvals(erad_log)
    ax.bar(xs_er, [e.get("n_treated", 0) for e in erad_log],
           color="#9467bd", alpha=0.8, width=_bar_width(xs_er))
    ax.set_ylabel("treated cells", fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    if use_dates:
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=30, ha="right")
    else:
        axes[-1].set_xlabel("timestep", fontsize=8)

    fig.suptitle("Population model — time series", fontsize=11)
    fig.tight_layout()

    out_path = Path(out_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Time series plot saved → %s", out_path)
    return out_path


def plot_animation(
    result: dict,
    out_dir: Path,
    filename: str = "population_animation.mp4",
    fps: int = 10,
    dpi: int = 120,
) -> Path:
    """
    Animate population density over all stored snapshots.

    Tries to save as MP4 (ffmpeg); falls back to GIF (pillow) if ffmpeg
    is not available.  Unsuitable habitat is shown in grey.

    Parameters
    ----------
    result : dict
        Return value of :meth:`PopulationModel.run`.
    out_dir : Path
        Output directory.
    filename : str
        Output filename (default ``"population_animation.mp4"``).
    fps : int
        Frames per second (default 10).
    dpi : int
        Resolution of each frame (default 120).

    Returns
    -------
    Path to the saved animation file.
    """
    from matplotlib.animation import FuncAnimation

    snaps        = result["density_snapshots"]   # (n_snaps, ny, nx)
    t_idx        = result["snapshot_timesteps"]  # list[int]
    lats         = result["lats"]
    lons         = result["lons"]
    tstamps      = result["timesteps"]
    habitat_mask = result.get("habitat_mask")    # (ny, nx) bool or None

    n_snaps = snaps.shape[0]
    if n_snaps == 0:
        raise ValueError("No snapshots to animate.")

    # Fixed colour scale across all frames
    if habitat_mask is not None:
        vmax = float(snaps[:, habitat_mask].max()) if snaps[:, habitat_mask].max() > 0 else 1.0
    else:
        vmax = float(snaps.max()) if snaps.max() > 0 else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = _make_density_cmap()

    # Build figure
    if _HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={"projection": proj})
    else:
        fig, ax = plt.subplots(figsize=(7, 5))

    # Draw the static unsuitable-habitat layer once
    _draw_unsuitable(ax, lons, lats, habitat_mask)

    # Initial density frame
    snap0 = _mask_snap(snaps[0], habitat_mask)
    if _HAS_CARTOPY:
        mesh = ax.pcolormesh(
            lons, lats, snap0, norm=norm, cmap=cmap,
            transform=ccrs.PlateCarree(), zorder=3,
        )
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "coastline", _CARTOPY_RESOLUTION,
                edgecolor="black", facecolor="none",
            ),
            linewidth=0.6, zorder=4,
        )
        ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]],
                      crs=ccrs.PlateCarree())
        ax.gridlines(linewidth=0.3, alpha=0.4)
    else:
        mesh = ax.pcolormesh(lons, lats, snap0, norm=norm, cmap=cmap, zorder=2)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    # Colourbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("density  [ind / m²]", fontsize=8)

    # Title updated each frame
    ts0 = t_idx[0]
    date0 = str(tstamps[ts0])[:10] if ts0 < len(tstamps) else f"t={ts0}"
    title = ax.set_title(f"t = {date0}", fontsize=10)

    def _update(frame: int):
        snap = _mask_snap(snaps[frame], habitat_mask)
        # pcolormesh expects a flat C array
        mesh.set_array(snap.ravel())
        ts = t_idx[frame]
        date_str = str(tstamps[ts])[:10] if ts < len(tstamps) else f"t={ts}"
        title.set_text(f"t = {date_str}")
        return mesh, title

    anim = FuncAnimation(
        fig, _update, frames=n_snaps, interval=1000 // fps, blit=True,
    )

    out_path = Path(out_dir) / filename
    try:
        anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
    except Exception:
        log.warning("ffmpeg unavailable — falling back to GIF")
        out_path = out_path.with_suffix(".gif")
        anim.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)

    plt.close(fig)
    log.info("Animation saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def _make_density_cmap() -> mcolors.LinearSegmentedColormap:
    """
    Inferno-based colormap with white at exactly 0.

    Zero-density suitable cells show as white; any positive value
    immediately picks up the inferno palette.
    """
    inferno = plt.cm.inferno
    colors = inferno(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]   # white for zero
    return mcolors.LinearSegmentedColormap.from_list(
        "density_cmap", colors, N=256,
    )


def _mask_snap(snap: np.ndarray, habitat_mask: np.ndarray | None) -> np.ndarray:
    """Return a copy of snap with unsuitable cells set to NaN."""
    out = snap.astype(float)
    if habitat_mask is not None:
        out[~habitat_mask] = np.nan
    return out


def _draw_unsuitable(ax, lons, lats, habitat_mask) -> None:
    """Fill unsuitable cells with a flat grey layer."""
    if habitat_mask is None:
        return
    grey_data = np.where(habitat_mask, np.nan, 0.0)
    grey_cmap = mcolors.ListedColormap([_UNSUITABLE_COLOR])
    kw = dict(zorder=1)
    if _HAS_CARTOPY:
        kw["transform"] = ccrs.PlateCarree()
    ax.pcolormesh(lons, lats, grey_data,
                  cmap=grey_cmap, vmin=-0.5, vmax=0.5, **kw)


def _draw_density_map(ax, snap, lons, lats, habitat_mask, norm, cmap) -> None:
    """Draw one density frame (unsuitable grey + density colourmap)."""
    _draw_unsuitable(ax, lons, lats, habitat_mask)

    masked = _mask_snap(snap, habitat_mask)
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
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)
    else:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    ax.pcolormesh(lons, lats, masked, **kw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar_width(xs: Sequence) -> float:
    """Estimate a sensible bar width for bar charts from x-axis values."""
    if len(xs) < 2:
        return 1.0
    x0, x1 = xs[0], xs[1]
    try:
        diff = (np.datetime64(x1) - np.datetime64(x0)) / np.timedelta64(1, "D")
        return float(diff) * 0.8
    except (TypeError, ValueError):
        return float(x1 - x0) * 0.8
