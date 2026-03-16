"""
run.py — orchestrator for the eradication model.

Configure the section below and run:
    python eradication/run.py
"""

import logging
from pathlib import Path

from eradication.connectivity.build import build_connectivity
from eradication.connectivity.load import load_connectivity
from eradication.habitat.suitability import build_habitat, load_habitat
from eradication.io.config import load_config, load_paths
from eradication.io.output import write_output
from eradication.population.model import PopulationModel

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configure run here
# ---------------------------------------------------------------------------
CONFIG_FILE          = Path("config/example_pacific_oyster.yaml")
REBUILD_CONNECTIVITY = False   # set True to force rerun of OceanTracker pipeline
REBUILD_HABITAT      = False   # set True to force rerun of habitat suitability
# ---------------------------------------------------------------------------

cfg   = load_config(CONFIG_FILE)
paths = load_paths()

region_dir = Path(paths["precomputed"]) / cfg["region"]
region_dir.mkdir(parents=True, exist_ok=True)

# --- Connectivity precomputation ---
connectivity_path = region_dir / "connectivity.npz"
if REBUILD_CONNECTIVITY or not connectivity_path.exists():
    log.info("Building connectivity tensor ...")
    build_connectivity(cfg, paths, region_dir)
else:
    log.info("Using existing connectivity: %s", connectivity_path)

# --- Habitat suitability map ---
habitat_path = region_dir / "habitat_suitability.nc"
if REBUILD_HABITAT or not habitat_path.exists():
    log.info("Building habitat suitability map ...")
    build_habitat(cfg, paths, region_dir)
else:
    log.info("Using existing habitat suitability: %s", habitat_path)

# --- Load precomputed data ---
connectivity = load_connectivity(connectivity_path)
habitat      = load_habitat(habitat_path)

# --- Run population model ---
log.info("Running population model ...")
model  = PopulationModel(cfg, connectivity, habitat)
result = model.run()

# --- Write output ---
log.info("Writing output ...")
output_path = region_dir / "output"
write_output(result, cfg, output_path)

log.info("Done. Output written to %s", output_path)
