from pathlib import Path

import numpy as np


def load_connectivity(
    path: Path,
    config: dict | None = None,
    region_dir: Path | None = None,
    habitat_mask: np.ndarray | None = None,
) -> dict:
    """Load a connectivity.npz written by lagrangian.run.build_connectivity.

    Returns a dict with keys: src_x, src_y, dst_x, dst_y, age, weight.
    Weights are decoded from uint8 back to float32 probabilities.

    Parameters
    ----------
    path : Path
        connectivity.npz file to load.
    config : dict, optional
        Full scenario config.  When provided, the ``debug.plot_sink_connectivity``
        flag is honoured and a debug PNG is written to *region_dir*.
    region_dir : Path, optional
        Output directory for the debug plot (required when *config* is given
        and the flag is set).
    habitat_mask : (ny, nx) bool array, optional
        Passed to the debug plot to shade unsuitable cells.
    """
    data = np.load(path)
    connectivity = {
        "src_x":  data["src_x"],
        "src_y":  data["src_y"],
        "dst_x":  data["dst_x"],
        "dst_y":  data["dst_y"],
        "age":    data["age"],
        "weight": data["weight"].astype(np.float32) / 255.0,
    }

    if config is not None and config.get("debug", {}).get("plot_sink_connectivity", False):
        from copernicus_pipeline.interpolate import make_model_grid
        from eradication.connectivity.plot import plot_sink_connectivity
        lons, lats = make_model_grid(config["spatial"])
        plot_sink_connectivity(
            connectivity,
            lons=lons,
            lats=lats,
            out_dir=region_dir,
            habitat_mask=habitat_mask,
        )

    return connectivity
