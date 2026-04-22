import numpy as np
from pathlib import Path

def idx_sort(path, name, N, reverse=False):
    scores = np.load(Path(path) / f"{name}.npy")
    if reverse:
        idx_sorted = np.argsort(scores)[::-1][:N]
    else:
        idx_sorted = np.argsort(scores)[:N]
    out_path = Path(path) / f"idx_sorted_{N}_{name}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, idx_sorted)
    return idx_sorted