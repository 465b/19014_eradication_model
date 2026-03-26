"""
eradication/habitat/suitability.py
-----------------------------------
Build and load the habitat suitability mask.

The mask is a boolean array on the eradication model grid:
  True  = cell is suitable habitat
  False = cell is not suitable habitat

Three sources of constraints are supported and combined with logical AND:

1. copernicus_data_based_constraints  — thresholds applied to variables in
   the pre-built bio_physical.nc (e.g. temperature, salinity).
2. user_data_based_constraints        — thresholds applied to user-supplied
   NetCDF files (e.g. substrate type, bathymetry).
3. mask_file                          — a pre-built boolean/integer mask
   NetCDF (2-D or 3-D).

If the config has no ``habitat`` section all cells are treated as suitable.

Public API
----------
build_habitat(config, paths, region_dir)  -> None  (writes habitat_suitability.nc)
load_habitat(habitat_path)                -> xr.DataArray  dims (lat,lon) or (time,lat,lon)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from copernicus_pipeline.interpolate import (
    build_model_timesteps,
    interp_to_model_grid,
    interp_to_model_timesteps,
    make_model_grid,
)

logger = logging.getLogger(__name__)

_HABITAT_FILENAME = "habitat_suitability.nc"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_habitat(config: dict, paths: dict, region_dir: Path) -> None:
    """Build the habitat suitability mask and write to region_dir/habitat_suitability.nc.

    Parameters
    ----------
    config : dict
        Full scenario config (from utils.config.load_config).
    paths : dict
        Paths config (from utils.config.load_paths) — currently unused but
        kept for API consistency with the other build_* functions.
    region_dir : Path
        Precomputed directory for this region.
    """
    region_dir = Path(region_dir)
    habitat_cfg = config.get("habitat")
    model_lons, model_lats = make_model_grid(config["spatial"])
    model_times = build_model_timesteps(config["temporal"])

    if habitat_cfg is None:
        logger.info("No 'habitat' section in config — all cells treated as suitable.")
        result = _all_true_2d(
            model_lons, model_lats,
            attrs={"description": "All cells suitable (no habitat constraints configured)"},
        )
        _write_habitat(result, region_dir)
        if config.get("debug", {}).get("plot_habitat", False):
            _plot_habitat_debug(result, region_dir)
        return

    layers: list[xr.DataArray] = []

    # 1. Copernicus data-based constraints
    cop_constraints = habitat_cfg.get("copernicus_data_based_constraints", [])
    if cop_constraints:
        bio_physical_path = region_dir / "copernicus" / "bio_physical.nc"
        if not bio_physical_path.exists():
            raise FileNotFoundError(
                f"bio_physical.nc not found at {bio_physical_path}. "
                "Run build_copernicus before build_habitat."
            )
        with xr.open_dataset(bio_physical_path) as bio_ds:
            for var_cfg in cop_constraints:
                layer = _copernicus_layer(var_cfg, bio_ds)
                layers.append(layer)
                _log_layer_stats("copernicus", var_cfg.get("variable", "?"), layer)

    # 2. User data-based constraints
    for layer_cfg in habitat_cfg.get("user_data_based_constraints", []):
        layer = _user_layer(layer_cfg, model_lons, model_lats, model_times)
        layers.append(layer)
        _log_layer_stats("user data", layer_cfg.get("variable", "?"), layer)

    # 3. Pre-built mask
    if "mask_file" in habitat_cfg:
        layer = _mask_layer(habitat_cfg, model_lons, model_lats, model_times)
        layers.append(layer)
        _log_layer_stats("mask file", habitat_cfg["mask_file"], layer)

    if not layers:
        logger.warning(
            "habitat section present but contains no constraints — all cells treated as suitable."
        )
        result = _all_true_2d(model_lons, model_lats)
    else:
        result = _stack_layers(layers)

    _write_habitat(result, region_dir)

    if config.get("debug", {}).get("plot_habitat", False):
        _plot_habitat_debug(result, region_dir)


def load_habitat(habitat_path: Path) -> xr.DataArray:
    """Load the precomputed habitat suitability mask.

    Returns
    -------
    xr.DataArray  dtype bool, dims (lat, lon) or (time, lat, lon)
    """
    ds = xr.open_dataset(habitat_path)
    return ds["habitat"].astype(bool)


# ---------------------------------------------------------------------------
# Layer builders
# ---------------------------------------------------------------------------


def _copernicus_layer(var_cfg: dict, bio_ds: xr.Dataset) -> xr.DataArray:
    """Apply min/max/equal_to constraints to a variable from bio_physical.nc.

    bio_physical.nc is already on the model grid and model timesteps, so no
    spatial or temporal interpolation is needed here.

    Returns
    -------
    xr.DataArray  bool, dims (time, lat, lon)
    """
    var_name = var_cfg["variable"]
    if var_name not in bio_ds:
        raise KeyError(
            f"Variable '{var_name}' not found in bio_physical.nc. "
            f"Available variables: {sorted(bio_ds.data_vars)}"
        )
    da = bio_ds[var_name]
    return _apply_constraint(
        da,
        min_val=var_cfg.get("min"),
        max_val=var_cfg.get("max"),
        equal_to=var_cfg.get("equal_to"),
    )


def _user_layer(
    layer_cfg: dict,
    model_lons: np.ndarray,
    model_lats: np.ndarray,
    model_times: np.ndarray,
) -> xr.DataArray:
    """Load a user-supplied NetCDF, interpolate to model grid, and apply constraints.

    If the source has a time dimension the data is also interpolated to model
    timesteps, producing a (time, lat, lon) result.  Without a time dimension
    the result has dims (lat, lon).

    Returns
    -------
    xr.DataArray  bool, dims (lat, lon) or (time, lat, lon)
    """
    path = Path(layer_cfg["path"])
    var_name = layer_cfg["variable"]

    if not path.exists():
        raise FileNotFoundError(f"User constraint file not found: {path}")

    with xr.open_dataset(path) as ds:
        if var_name not in ds:
            raise KeyError(
                f"Variable '{var_name}' not found in {path}. "
                f"Available variables: {sorted(ds.data_vars)}"
            )
        da = ds[var_name].load()  # load into memory before dataset closes

    da_on_grid = _interp_to_grid(da, model_lons, model_lats, model_times, var_name)
    return _apply_constraint(
        da_on_grid,
        min_val=layer_cfg.get("min"),
        max_val=layer_cfg.get("max"),
        equal_to=layer_cfg.get("equal_to"),
    )


def _mask_layer(
    habitat_cfg: dict,
    model_lons: np.ndarray,
    model_lats: np.ndarray,
    model_times: np.ndarray,
) -> xr.DataArray:
    """Load a pre-built boolean/integer mask NetCDF and interpolate to model grid.

    Supports both 2-D (static) and 3-D (time-varying) mask files.
    Bilinear interpolation is used; a 0.5 threshold converts the result to
    boolean (equivalent to nearest-neighbour for clean 0/1 masks).

    Returns
    -------
    xr.DataArray  bool, dims (lat, lon) or (time, lat, lon)
    """
    path = Path(habitat_cfg["mask_file"])
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    with xr.open_dataset(path) as ds:
        if "mask_variable" in habitat_cfg:
            var_name = habitat_cfg["mask_variable"]
            if var_name not in ds:
                raise KeyError(
                    f"Variable '{var_name}' not found in mask file {path}. "
                    f"Available variables: {sorted(ds.data_vars)}"
                )
        else:
            var_name = next(iter(ds.data_vars))
        da = ds[var_name].load()

    da_on_grid = _interp_to_grid(da, model_lons, model_lats, model_times, var_name)
    # Threshold at 0.5 handles bilinear interpolation artefacts for integer masks.
    return da_on_grid > 0.5


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _interp_to_grid(
    da: xr.DataArray,
    model_lons: np.ndarray,
    model_lats: np.ndarray,
    model_times: np.ndarray,
    var_name: str = "",
) -> xr.DataArray:
    """Interpolate *da* onto the model spatial grid (and timesteps if time-varying).

    Parameters
    ----------
    da : xr.DataArray
        Source data with spatial dimensions (lat/lon or equivalents).
        May optionally have a ``time`` dimension.
    model_lons, model_lats : np.ndarray
        1-D model grid cell centres.
    model_times : np.ndarray of np.datetime64
        Model timestep array (used only when *da* has a time dimension).
    var_name : str
        For logging.

    Returns
    -------
    xr.DataArray  dims (lat, lon) for static inputs or (time, lat, lon) for
    time-varying inputs.
    """
    if "time" in da.dims:
        da_grid = interp_to_model_grid(da, model_lons, model_lats, var_name=var_name)
        return interp_to_model_timesteps(da_grid, model_times, var_name=var_name)

    # Static input: add a dummy time dim so interp_to_model_grid can handle it,
    # then drop it afterwards.
    dummy_time = np.array(["2000-01-01"], dtype="datetime64[D]")
    da_with_time = da.expand_dims({"time": dummy_time})
    da_grid = interp_to_model_grid(da_with_time, model_lons, model_lats, var_name=var_name)
    return da_grid.isel(time=0, drop=True)


def _apply_constraint(
    da: xr.DataArray,
    min_val: float | None,
    max_val: float | None,
    equal_to: object | None,
) -> xr.DataArray:
    """Return a boolean DataArray: True where *da* satisfies all given constraints.

    NaN values in *da* are always treated as not suitable (comparisons return
    False for NaN, following standard numpy behaviour).

    Parameters
    ----------
    da : xr.DataArray
    min_val : float | None
        Inclusive lower bound.
    max_val : float | None
        Inclusive upper bound.
    equal_to : scalar | None
        Exact match value — useful for integer-coded categories or booleans.
        For string-typed arrays a UserWarning is issued.
    """
    mask = xr.ones_like(da, dtype=bool)

    if min_val is not None:
        mask = mask & (da >= float(min_val))
    if max_val is not None:
        mask = mask & (da <= float(max_val))
    if equal_to is not None:
        if da.dtype.kind in ("U", "S", "O"):
            warnings.warn(
                f"equal_to used on a string-typed variable '{da.name}'. "
                "Integer-coded categories are recommended for reliable NetCDF comparisons.",
                UserWarning,
                stacklevel=4,
            )
        mask = mask & (da == equal_to)

    return mask


def _stack_layers(layers: list[xr.DataArray]) -> xr.DataArray:
    """Combine layers with logical AND, broadcasting 2-D layers over time as needed.

    xarray broadcasting rules ensure that a (lat, lon) layer is broadcast
    correctly against a (time, lat, lon) layer.
    """
    result = layers[0]
    for layer in layers[1:]:
        result = result & layer
    return result.rename("habitat")


def _all_true_2d(
    model_lons: np.ndarray,
    model_lats: np.ndarray,
    attrs: dict | None = None,
) -> xr.DataArray:
    return xr.DataArray(
        np.ones((len(model_lats), len(model_lons)), dtype=bool),
        dims=["lat", "lon"],
        coords={"lat": model_lats, "lon": model_lons},
        name="habitat",
        attrs=attrs or {},
    )


def _write_habitat(mask: xr.DataArray, region_dir: Path) -> None:
    out_path = region_dir / _HABITAT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    da = mask.rename("habitat")
    if "description" not in da.attrs:
        da.attrs["description"] = (
            "Habitat suitability mask — True (1) = suitable, False (0) = unsuitable."
        )

    ds = da.to_dataset()
    # Encode bool as uint8; some NetCDF drivers do not support bool dtype directly.
    encoding = {"habitat": {"dtype": "u1"}}
    ds.to_netcdf(out_path, encoding=encoding)
    logger.info("Habitat suitability written to %s", out_path)


def _plot_habitat_debug(mask: xr.DataArray, region_dir: Path) -> None:
    """Save a debug PNG of the habitat suitability mask.

    For a 3-D (time, lat, lon) mask the time-mean fraction of suitable
    timesteps is plotted [0, 1].  For a 2-D (lat, lon) mask the boolean
    values are plotted directly.

    Saved to: region_dir/habitat_suitability_debug.png
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    lons = mask["lon"].values
    lats = mask["lat"].values

    if mask.ndim == 3:
        data2d = mask.values.astype(float).mean(axis=0)   # fraction [0, 1]
        cbar_label = "Fraction of timesteps suitable  [0 = never, 1 = always]"
        title = "Habitat suitability  —  time-mean fraction suitable"
        cmap = "RdYlGn"
        vmin, vmax = 0.0, 1.0
    else:
        data2d = mask.values.astype(float)
        cbar_label = "Suitable  [0 = no, 1 = yes]"
        title = "Habitat suitability mask"
        cmap = "RdYlGn"
        vmin, vmax = 0.0, 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    pad_lon = max((lons[-1] - lons[0]) * 0.5, 0.05)
    pad_lat = max((lats[-1] - lats[0]) * 0.5, 0.05)
    extent = [
        lons[0]  - pad_lon, lons[-1] + pad_lon,
        lats[0]  - pad_lat, lats[-1] + pad_lat,
    ]

    fig, ax = plt.subplots(
        figsize=(7, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="wheat", alpha=0.6, zorder=1)

    mesh = ax.pcolormesh(
        lons, lats, data2d,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        zorder=0,
    )
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.85, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    fig.suptitle(title, fontsize=11)

    out = region_dir / "habitat_suitability_debug.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Habitat debug plot saved → %s", out)


def _log_layer_stats(source: str, name: str, layer: xr.DataArray) -> None:
    total = layer.shape[-1] * layer.shape[-2]
    if layer.ndim == 3:
        suitable = int(layer.values.any(axis=0).sum())
        logger.info(
            "  [%s] '%s': %d / %d cells suitable in at least one timestep.",
            source, name, suitable, total,
        )
    else:
        suitable = int(layer.values.sum())
        logger.info("  [%s] '%s': %d / %d cells suitable.", source, name, suitable, total)
