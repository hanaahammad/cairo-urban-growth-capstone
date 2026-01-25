import numpy as np
import pandas as pd

def extract_patch_features(t0, built0, patch=5):
    """
    Create a patch dataset:
    For each grid cell, compute local neighborhood features.
    Label = built0 (built-up at t0)
    """
    h, w = t0.shape
    pad = patch // 2

    # padding
    padded = np.pad(t0, pad_width=pad, mode="reflect")
    padded_b = np.pad(built0, pad_width=pad, mode="reflect")

    rows = []
    for i in range(h):
        for j in range(w):
            block = padded[i:i+patch, j:j+patch]
            block_b = padded_b[i:i+patch, j:j+patch]

            rows.append({
                "row": i,
                "col": j,
                "mean_intensity": float(block.mean()),
                "std_intensity": float(block.std()),
                "min_intensity": float(block.min()),
                "max_intensity": float(block.max()),
                "built_neighbors_ratio": float(block_b.mean()),
                "label": int(built0[i, j]),
            })

    return pd.DataFrame(rows)
