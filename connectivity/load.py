from pathlib import Path

import numpy as np


def load_connectivity(path: Path) -> dict:
    """Load a connectivity.npz written by lagrangian.run.build_connectivity.

    Returns a dict with keys: src_x, src_y, dst_x, dst_y, age, weight.
    Weights are decoded from uint8 back to float32 probabilities.
    """
    data = np.load(path)
    return {
        "src_x":  data["src_x"],
        "src_y":  data["src_y"],
        "dst_x":  data["dst_x"],
        "dst_y":  data["dst_y"],
        "age":    data["age"],
        "weight": data["weight"].astype(np.float32) / 255.0,
    }
